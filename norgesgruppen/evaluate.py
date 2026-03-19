"""Local mAP evaluation matching the NM i AI competition scoring.

Score = 0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5

Detection mAP: IoU >= 0.5, category IGNORED (all predictions treated as one class)
Classification mAP: IoU >= 0.5 AND correct category_id

Usage:
    # Evaluate existing predictions
    python evaluate.py --predictions predictions.json --annotations data/train/annotations.json

    # Run pipeline end-to-end and evaluate
    python evaluate.py --run-pipeline --model-dir . --images data/train/images \
        --annotations data/train/annotations.json

    # Quick validation on subset
    python evaluate.py --predictions predictions.json --annotations data/train/annotations.json \
        --max-images 20
"""

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def load_coco_gt(annotations_path: str, max_images: int | None = None) -> COCO:
    """Load ground truth annotations as a COCO object."""
    with open(annotations_path) as f:
        data = json.load(f)

    if max_images is not None:
        # Subset to first N images
        image_ids = sorted(img["id"] for img in data["images"])[:max_images]
        image_id_set = set(image_ids)
        data["images"] = [img for img in data["images"] if img["id"] in image_id_set]
        data["annotations"] = [
            ann for ann in data["annotations"] if ann["image_id"] in image_id_set
        ]

    # Write to temp file for COCO API (it needs a file path)
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, tmp)
    tmp.close()

    coco_gt = COCO(tmp.name)
    Path(tmp.name).unlink()
    return coco_gt, data.get("categories", [])


def load_predictions(predictions_path: str, image_ids: set | None = None) -> list:
    """Load predictions and optionally filter to specific image IDs."""
    with open(predictions_path) as f:
        preds = json.load(f)

    if image_ids is not None:
        preds = [p for p in preds if p["image_id"] in image_ids]

    return preds


def make_detection_only(gt_data_path: str, predictions: list, max_images: int | None = None):
    """Create single-class versions for detection-only mAP.

    For detection mAP, category is ignored: all boxes become category 1.
    Returns (coco_gt_det, predictions_det).
    """
    with open(gt_data_path) as f:
        data = json.load(f)

    if max_images is not None:
        image_ids = sorted(img["id"] for img in data["images"])[:max_images]
        image_id_set = set(image_ids)
        data["images"] = [img for img in data["images"] if img["id"] in image_id_set]
        data["annotations"] = [
            ann for ann in data["annotations"] if ann["image_id"] in image_id_set
        ]

    # Replace all categories with a single "object" category
    data["categories"] = [{"id": 1, "name": "object", "supercategory": "object"}]

    # Set all GT annotations to category 1
    for ann in data["annotations"]:
        ann["category_id"] = 1

    # Ensure all annotations have 'iscrowd' and 'area'
    for ann in data["annotations"]:
        if "area" not in ann:
            x, y, w, h = ann["bbox"]
            ann["area"] = w * h
        if "iscrowd" not in ann:
            ann["iscrowd"] = 0

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, tmp)
    tmp.close()

    coco_gt_det = COCO(tmp.name)
    Path(tmp.name).unlink()

    # Set all predictions to category 1
    preds_det = []
    for p in predictions:
        preds_det.append({
            "image_id": p["image_id"],
            "category_id": 1,
            "bbox": p["bbox"],
            "score": p["score"],
        })

    return coco_gt_det, preds_det


def ensure_ann_fields(annotations_path: str, max_images: int | None = None):
    """Load annotations and ensure area/iscrowd fields exist for COCOeval."""
    with open(annotations_path) as f:
        data = json.load(f)

    if max_images is not None:
        image_ids = sorted(img["id"] for img in data["images"])[:max_images]
        image_id_set = set(image_ids)
        data["images"] = [img for img in data["images"] if img["id"] in image_id_set]
        data["annotations"] = [
            ann for ann in data["annotations"] if ann["image_id"] in image_id_set
        ]

    for ann in data["annotations"]:
        if "area" not in ann:
            x, y, w, h = ann["bbox"]
            ann["area"] = w * h
        if "iscrowd" not in ann:
            ann["iscrowd"] = 0

    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False)
    json.dump(data, tmp)
    tmp.close()

    coco_gt = COCO(tmp.name)
    Path(tmp.name).unlink()
    return coco_gt, data.get("categories", [])


def run_coco_eval(coco_gt: COCO, predictions: list, iou_thr: float = 0.5) -> dict:
    """Run COCOeval and return mAP at the specified IoU threshold.

    Returns dict with 'mAP', 'per_class_ap', and summary stats.
    """
    if not predictions:
        return {"mAP": 0.0, "per_class_ap": {}, "precision": None, "recall": None}

    coco_dt = coco_gt.loadRes(predictions)

    coco_eval = COCOeval(coco_gt, coco_dt, "bbox")
    coco_eval.params.iouThrs = np.array([iou_thr])
    coco_eval.params.maxDets = [1, 10, 500]

    coco_eval.evaluate()
    coco_eval.accumulate()

    # Extract mAP at the single IoU threshold
    # precision shape: [T, R, K, A, M] = [iou_thrs, recall_thrs, cat_ids, area_rngs, max_dets]
    # We want T=0 (our single threshold), A=0 (all areas), M=-1 (max maxDets)
    precision = coco_eval.eval["precision"]  # [T, R, K, A, M]

    # mAP across all categories: mean over R and K dimensions at T=0, A=0, M=-1
    # precision[0, :, :, 0, -1] -> [R, K]
    per_class_precision = precision[0, :, :, 0, -1]  # [R, K]

    cat_ids = coco_eval.params.catIds
    per_class_ap = {}
    for ki, cat_id in enumerate(cat_ids):
        ap_vals = per_class_precision[:, ki]
        if ap_vals[ap_vals > -1].size > 0:
            per_class_ap[cat_id] = float(np.mean(ap_vals[ap_vals > -1]))
        else:
            per_class_ap[cat_id] = 0.0

    valid_aps = [v for v in per_class_ap.values() if v > 0]
    mean_ap = float(np.mean(list(per_class_ap.values()))) if per_class_ap else 0.0

    # Also get recall
    recall = coco_eval.eval["recall"]  # [T, K, A, M]

    return {
        "mAP": mean_ap,
        "per_class_ap": per_class_ap,
        "precision_curve": precision,
        "recall_curve": recall,
        "n_classes_with_ap": len(valid_aps),
        "n_classes_total": len(cat_ids),
    }


def print_results(
    det_results: dict,
    cls_results: dict,
    categories: list,
    predictions: list,
    n_gt_anns: int,
    n_gt_images: int,
):
    """Print detailed evaluation results."""
    det_map = det_results["mAP"]
    cls_map = cls_results["mAP"]
    combined = 0.7 * det_map + 0.3 * cls_map

    cat_lookup = {c["id"]: c["name"] for c in categories}

    print()
    print("=" * 70)
    print("  NM i AI — NorgesGruppen Evaluation Results")
    print("=" * 70)
    print()
    print(f"  Combined Score:           {combined:.4f}")
    print(f"    = 0.7 * {det_map:.4f} (detection) + 0.3 * {cls_map:.4f} (classification)")
    print()
    print(f"  Detection mAP@0.5:       {det_map:.4f}  (category ignored)")
    print(f"  Classification mAP@0.5:  {cls_map:.4f}  (category must match)")
    print()
    print("-" * 70)
    print(f"  Ground truth:  {n_gt_anns} boxes across {n_gt_images} images")
    print(f"  Predictions:   {len(predictions)} boxes")
    pred_images = len(set(p["image_id"] for p in predictions)) if predictions else 0
    print(f"  Images w/ predictions: {pred_images}")
    print()

    # Detection stats
    if det_results.get("n_classes_total"):
        print(f"  Detection:  {det_results['n_classes_with_ap']}/{det_results['n_classes_total']}"
              f" classes with AP > 0")

    # Classification stats
    if cls_results.get("n_classes_total"):
        print(f"  Classification:  {cls_results['n_classes_with_ap']}/{cls_results['n_classes_total']}"
              f" classes with AP > 0")

    # Per-class AP for classification (best and worst)
    cls_ap = cls_results.get("per_class_ap", {})
    if cls_ap:
        print()
        print("-" * 70)
        print("  BEST classes (classification AP@0.5)")
        print(f"  {'Category':<45} {'AP':>8}")
        print("  " + "-" * 55)

        sorted_ap = sorted(cls_ap.items(), key=lambda x: x[1], reverse=True)
        for cat_id, ap in sorted_ap[:15]:
            name = cat_lookup.get(cat_id, f"id={cat_id}")
            if len(name) > 43:
                name = name[:41] + ".."
            print(f"  {name:<45} {ap:>8.4f}")

        print()
        print("  WORST classes (classification AP@0.5, AP > 0)")
        print(f"  {'Category':<45} {'AP':>8}")
        print("  " + "-" * 55)

        worst = [(cid, ap) for cid, ap in sorted_ap if ap > 0]
        for cat_id, ap in worst[-15:]:
            name = cat_lookup.get(cat_id, f"id={cat_id}")
            if len(name) > 43:
                name = name[:41] + ".."
            print(f"  {name:<45} {ap:>8.4f}")

        # Classes with AP = 0
        zero_ap = [cid for cid, ap in cls_ap.items() if ap == 0.0]
        if zero_ap:
            print()
            print(f"  Classes with AP = 0: {len(zero_ap)} / {len(cls_ap)}")

    print()
    print("=" * 70)

    # Score breakdown for quick copy-paste
    print()
    print(f"SCORE: {combined:.4f}  (det={det_map:.4f}, cls={cls_map:.4f})")
    print()


def run_pipeline(model_dir: str, images_dir: str, output_path: str):
    """Run run.py to generate predictions."""
    run_script = Path(model_dir) / "run.py"
    if not run_script.exists():
        print(f"Error: {run_script} not found")
        sys.exit(1)

    cmd = [
        sys.executable,
        str(run_script),
        "--input", str(images_dir),
        "--output", str(output_path),
    ]

    print(f"Running pipeline: {' '.join(cmd)}")
    print("-" * 70)
    result = subprocess.run(cmd, capture_output=False)
    print("-" * 70)

    if result.returncode != 0:
        print(f"Pipeline failed with return code {result.returncode}")
        sys.exit(1)

    print(f"Pipeline complete. Predictions at: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate object detection predictions using competition scoring"
    )
    parser.add_argument(
        "--predictions", type=str,
        help="Path to predictions.json"
    )
    parser.add_argument(
        "--annotations", type=str, required=True,
        help="Path to COCO annotations.json (ground truth)"
    )
    parser.add_argument(
        "--max-images", type=int, default=None,
        help="Evaluate on first N images only (for quick validation)"
    )
    parser.add_argument(
        "--run-pipeline", action="store_true",
        help="Run run.py first to generate predictions"
    )
    parser.add_argument(
        "--model-dir", type=str, default=".",
        help="Directory containing run.py and model weights"
    )
    parser.add_argument(
        "--images", type=str,
        help="Image directory (required with --run-pipeline)"
    )
    args = parser.parse_args()

    annotations_path = str(Path(args.annotations).resolve())

    # Run pipeline if requested
    if args.run_pipeline:
        if not args.images:
            print("Error: --images required with --run-pipeline")
            sys.exit(1)

        output_path = str(Path(args.model_dir) / "eval_predictions.json")
        run_pipeline(args.model_dir, args.images, output_path)
        predictions_path = output_path
    elif args.predictions:
        predictions_path = str(Path(args.predictions).resolve())
    else:
        print("Error: provide --predictions or --run-pipeline")
        sys.exit(1)

    print(f"\nAnnotations: {annotations_path}")
    print(f"Predictions: {predictions_path}")
    if args.max_images:
        print(f"Max images:  {args.max_images}")

    # Load GT with proper fields
    coco_gt_cls, categories = ensure_ann_fields(annotations_path, args.max_images)
    gt_image_ids = set(coco_gt_cls.getImgIds())
    n_gt_images = len(gt_image_ids)
    n_gt_anns = len(coco_gt_cls.getAnnIds())

    # Load predictions
    predictions = load_predictions(predictions_path, gt_image_ids)
    print(f"\nLoaded {len(predictions)} predictions for {n_gt_images} GT images")

    if not predictions:
        print("\nNo predictions found. Score = 0.0")
        return

    # --- Classification mAP (IoU >= 0.5, category must match) ---
    print("\nComputing classification mAP@0.5 ...")
    cls_results = run_coco_eval(coco_gt_cls, predictions, iou_thr=0.5)

    # --- Detection mAP (IoU >= 0.5, category ignored — single class) ---
    print("Computing detection mAP@0.5 ...")
    coco_gt_det, preds_det = make_detection_only(
        annotations_path, predictions, args.max_images
    )
    det_results = run_coco_eval(coco_gt_det, preds_det, iou_thr=0.5)

    # --- Print results ---
    print_results(
        det_results, cls_results, categories,
        predictions, n_gt_anns, n_gt_images,
    )


if __name__ == "__main__":
    main()
