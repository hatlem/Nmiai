"""Local evaluation — compute detection mAP and classification mAP.

Mirrors the competition scoring:
  Score = 0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5

Usage:
    python eval_local.py --predictions predictions.json --annotations data/train/annotations.json

    # Or run the full pipeline: detect + classify + evaluate
    python eval_local.py --run-pipeline --model best.pt --input data/train/images --annotations data/train/annotations.json
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from collections import defaultdict

import numpy as np


def compute_iou(box1, box2):
    """Compute IoU between two boxes in [x, y, w, h] format."""
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2

    xa = max(x1, x2)
    ya = max(y1, y2)
    xb = min(x1 + w1, x2 + w2)
    yb = min(y1 + h1, y2 + h2)

    inter = max(0, xb - xa) * max(0, yb - ya)
    union = w1 * h1 + w2 * h2 - inter

    return inter / union if union > 0 else 0


def compute_ap(recalls, precisions):
    """Compute AP using 101-point interpolation (COCO style)."""
    mrec = np.concatenate(([0.0], recalls, [1.0]))
    mpre = np.concatenate(([1.0], precisions, [0.0]))

    # Make precision monotonically decreasing
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])

    # 101-point interpolation
    recall_points = np.linspace(0, 1, 101)
    ap = 0.0
    for r in recall_points:
        prec_at_r = mpre[mrec >= r]
        ap += (prec_at_r[0] if len(prec_at_r) > 0 else 0.0)

    return ap / 101


def evaluate_predictions(predictions, annotations, iou_threshold=0.5):
    """Compute detection mAP and classification mAP.

    Detection: IoU >= threshold, category ignored
    Classification: IoU >= threshold AND category matches
    """
    # Group ground truth by image_id
    gt_by_image = defaultdict(list)
    for ann in annotations["annotations"]:
        gt_by_image[ann["image_id"]].append({
            "bbox": ann["bbox"],
            "category_id": ann["category_id"],
        })

    # Group predictions by image_id, sorted by score descending
    pred_by_image = defaultdict(list)
    for pred in predictions:
        pred_by_image[pred["image_id"]].append(pred)

    for img_id in pred_by_image:
        pred_by_image[img_id].sort(key=lambda x: -x["score"])

    # --- Detection mAP (category-agnostic) ---
    all_det_scores = []
    all_det_tp = []
    total_gt = sum(len(gts) for gts in gt_by_image.values())

    all_preds_sorted = sorted(predictions, key=lambda x: -x["score"])

    gt_matched_det = defaultdict(set)  # image_id -> set of matched gt indices

    for pred in all_preds_sorted:
        img_id = pred["image_id"]
        gts = gt_by_image.get(img_id, [])

        best_iou = 0
        best_gt_idx = -1

        for gt_idx, gt in enumerate(gts):
            iou = compute_iou(pred["bbox"], gt["bbox"])
            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        is_tp = (best_iou >= iou_threshold and best_gt_idx not in gt_matched_det[img_id])

        if is_tp:
            gt_matched_det[img_id].add(best_gt_idx)

        all_det_scores.append(pred["score"])
        all_det_tp.append(1 if is_tp else 0)

    # Compute detection AP
    det_tp_cumsum = np.cumsum(all_det_tp)
    det_fp_cumsum = np.cumsum([1 - tp for tp in all_det_tp])
    det_recalls = det_tp_cumsum / total_gt if total_gt > 0 else det_tp_cumsum
    det_precisions = det_tp_cumsum / (det_tp_cumsum + det_fp_cumsum)

    det_mAP = compute_ap(det_recalls, det_precisions)

    # --- Classification mAP (per category) ---
    # Group by category
    categories = set(ann["category_id"] for ann in annotations["annotations"])

    cat_aps = []
    for cat_id in sorted(categories):
        cat_gts = {}  # image_id -> list of gt boxes for this category
        cat_total_gt = 0
        for img_id, gts in gt_by_image.items():
            cat_boxes = [gt for gt in gts if gt["category_id"] == cat_id]
            if cat_boxes:
                cat_gts[img_id] = cat_boxes
                cat_total_gt += len(cat_boxes)

        if cat_total_gt == 0:
            continue

        # Get predictions for this category
        cat_preds = [p for p in predictions if p["category_id"] == cat_id]
        cat_preds.sort(key=lambda x: -x["score"])

        tp_list = []
        gt_matched = defaultdict(set)

        for pred in cat_preds:
            img_id = pred["image_id"]
            gts = cat_gts.get(img_id, [])

            best_iou = 0
            best_idx = -1
            for idx, gt in enumerate(gts):
                iou = compute_iou(pred["bbox"], gt["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx

            is_tp = (best_iou >= iou_threshold and best_idx not in gt_matched[img_id])
            if is_tp:
                gt_matched[img_id].add(best_idx)

            tp_list.append(1 if is_tp else 0)

        if not tp_list:
            cat_aps.append(0.0)
            continue

        tp_cumsum = np.cumsum(tp_list)
        fp_cumsum = np.cumsum([1 - tp for tp in tp_list])
        recalls = tp_cumsum / cat_total_gt
        precisions = tp_cumsum / (tp_cumsum + fp_cumsum)

        ap = compute_ap(recalls, precisions)
        cat_aps.append(ap)

    cls_mAP = np.mean(cat_aps) if cat_aps else 0.0

    # Combined score
    final_score = 0.7 * det_mAP + 0.3 * cls_mAP

    return {
        "detection_mAP": det_mAP,
        "classification_mAP": cls_mAP,
        "final_score": final_score,
        "total_predictions": len(predictions),
        "total_gt": total_gt,
        "categories_evaluated": len(cat_aps),
        "det_recall_at_max": float(det_recalls[-1]) if len(det_recalls) > 0 else 0,
        "det_precision_at_max": float(det_precisions[-1]) if len(det_precisions) > 0 else 0,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", help="Path to predictions.json")
    parser.add_argument("--annotations", default="data/train/annotations.json")
    parser.add_argument("--run-pipeline", action="store_true", help="Run full pipeline first")
    parser.add_argument("--model", default="best.pt", help="Model path (for --run-pipeline)")
    parser.add_argument("--input", default="data/train/images", help="Images dir (for --run-pipeline)")
    args = parser.parse_args()

    if args.run_pipeline:
        print("Running pipeline...")
        pred_path = "/tmp/eval_predictions.json"
        cmd = [sys.executable, "run.py",
               "--input", args.input, "--output", pred_path]
        print(f"  Command: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        print(result.stdout)
        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            return
        args.predictions = pred_path

    if not args.predictions:
        parser.error("--predictions is required (or use --run-pipeline)")

    with open(args.predictions) as f:
        predictions = json.load(f)
    with open(args.annotations) as f:
        annotations = json.load(f)

    print(f"Predictions: {len(predictions)}")
    print(f"Ground truth: {len(annotations['annotations'])} boxes in {len(annotations['images'])} images")
    print()

    results = evaluate_predictions(predictions, annotations)

    print("=" * 50)
    print(f"  Detection mAP@0.5:       {results['detection_mAP']:.4f}")
    print(f"  Classification mAP@0.5:  {results['classification_mAP']:.4f}")
    print(f"  ---")
    print(f"  Final Score:             {results['final_score']:.4f}")
    print(f"  (= 0.7 × {results['detection_mAP']:.4f} + 0.3 × {results['classification_mAP']:.4f})")
    print("=" * 50)
    print(f"  Total predictions: {results['total_predictions']}")
    print(f"  Total GT boxes:   {results['total_gt']}")
    print(f"  Categories eval:  {results['categories_evaluated']}")
    print(f"  Det recall:       {results['det_recall_at_max']:.3f}")
    print(f"  Det precision:    {results['det_precision_at_max']:.3f}")


if __name__ == "__main__":
    main()
