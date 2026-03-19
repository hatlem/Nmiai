"""Visualize predictions vs ground truth on training images.

Three modes:
  1. Ground truth only — see what annotations look like
  2. Predictions only — see what model outputs
  3. Side-by-side — GT (green) vs predictions (red/blue) overlay

Supports both COCO annotations.json and predictions.json formats.

Usage:
    # View ground truth annotations
    python visualize.py --mode gt --images data/train/images --annotations data/train/annotations.json

    # View model predictions
    python visualize.py --mode pred --images data/train/images --predictions predictions.json

    # Side-by-side comparison (GT green, correct pred blue, wrong pred red)
    python visualize.py --mode compare --images data/train/images \
        --annotations data/train/annotations.json --predictions predictions.json

    # Single image deep-dive
    python visualize.py --mode compare --images data/train/images \
        --annotations data/train/annotations.json --predictions predictions.json \
        --image-id 42

    # Save to file instead of displaying
    python visualize.py --mode gt --images data/train/images \
        --annotations data/train/annotations.json --output viz_output/
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


# Colors (BGR)
GREEN = (0, 200, 0)       # Ground truth
BLUE = (255, 150, 0)      # Correct prediction (IoU >= 0.5 and correct class)
RED = (0, 0, 255)         # Wrong prediction (missed class or false positive)
YELLOW = (0, 230, 230)    # Missed GT (false negative)
WHITE = (255, 255, 255)
GRAY = (180, 180, 180)


def load_annotations(path: str) -> dict:
    """Load COCO annotations.json and index by image_id."""
    with open(path) as f:
        data = json.load(f)

    categories = {c["id"]: c["name"] for c in data.get("categories", [])}
    images = {img["id"]: img for img in data.get("images", [])}

    # Group annotations by image_id
    anns_by_image = {}
    for ann in data.get("annotations", []):
        img_id = ann["image_id"]
        anns_by_image.setdefault(img_id, []).append(ann)

    return {"categories": categories, "images": images, "anns_by_image": anns_by_image}


def load_predictions(path: str) -> dict:
    """Load predictions.json and index by image_id."""
    with open(path) as f:
        preds = json.load(f)

    preds_by_image = {}
    for pred in preds:
        img_id = pred["image_id"]
        preds_by_image.setdefault(img_id, []).append(pred)

    return preds_by_image


def iou(box_a, box_b):
    """IoU between two COCO boxes [x, y, w, h]."""
    ax, ay, aw, ah = box_a
    bx, by, bw, bh = box_b

    x1 = max(ax, bx)
    y1 = max(ay, by)
    x2 = min(ax + aw, bx + bw)
    y2 = min(ay + ah, by + bh)

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = aw * ah
    area_b = bw * bh
    return inter / (area_a + area_b - inter)


def draw_box(img, bbox, color, label="", thickness=2, font_scale=0.4):
    """Draw a bounding box with optional label."""
    x, y, w, h = [int(v) for v in bbox]
    cv2.rectangle(img, (x, y), (x + w, y + h), color, thickness)

    if label:
        # Background for text
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 4, y), color, -1)
        cv2.putText(img, label, (x + 2, y - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, WHITE, 1, cv2.LINE_AA)


def shorten_name(name: str, max_len: int = 20) -> str:
    """Shorten product name for display."""
    if len(name) <= max_len:
        return name
    return name[:max_len - 2] + ".."


def match_predictions_to_gt(gt_anns, pred_anns, iou_threshold=0.5):
    """Match predictions to GT boxes. Returns matched, unmatched_pred, unmatched_gt."""
    matched = []        # (gt, pred, iou, class_correct)
    unmatched_preds = list(pred_anns)
    unmatched_gts = list(gt_anns)

    # Sort predictions by score descending
    pred_sorted = sorted(pred_anns, key=lambda p: p.get("score", 0), reverse=True)
    gt_available = list(range(len(gt_anns)))

    pred_matched_idx = set()

    for pi, pred in enumerate(pred_sorted):
        best_iou = 0
        best_gi = -1

        for gi in gt_available:
            gt = gt_anns[gi]
            cur_iou = iou(gt["bbox"], pred["bbox"])
            if cur_iou > best_iou:
                best_iou = cur_iou
                best_gi = gi

        if best_iou >= iou_threshold and best_gi >= 0:
            gt = gt_anns[best_gi]
            class_correct = gt["category_id"] == pred["category_id"]
            matched.append((gt, pred, best_iou, class_correct))
            gt_available.remove(best_gi)
            pred_matched_idx.add(pi)

    unmatched_preds = [p for i, p in enumerate(pred_sorted) if i not in pred_matched_idx]
    unmatched_gts = [gt_anns[i] for i in gt_available]

    return matched, unmatched_preds, unmatched_gts


def visualize_gt(img, anns, categories, max_labels=50):
    """Draw ground truth annotations in green."""
    vis = img.copy()
    for i, ann in enumerate(anns):
        name = categories.get(ann["category_id"], f"id={ann['category_id']}")
        label = shorten_name(name) if i < max_labels else ""
        draw_box(vis, ann["bbox"], GREEN, label)

    # Stats overlay
    text = f"GT: {len(anns)} boxes"
    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, GREEN, 2)
    return vis


def visualize_pred(img, preds, categories, conf_threshold=0.1, max_labels=50):
    """Draw predictions with confidence coloring."""
    vis = img.copy()
    shown = 0
    for pred in sorted(preds, key=lambda p: p.get("score", 0), reverse=True):
        score = pred.get("score", 0)
        if score < conf_threshold:
            continue

        cat_id = pred["category_id"]
        name = categories.get(cat_id, f"id={cat_id}")

        # Color by confidence
        if score > 0.7:
            color = BLUE
        elif score > 0.3:
            color = YELLOW
        else:
            color = RED

        label = f"{shorten_name(name)} {score:.2f}" if shown < max_labels else ""
        draw_box(vis, pred["bbox"], color, label)
        shown += 1

    text = f"Pred: {shown} boxes (>{conf_threshold:.2f})"
    cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, BLUE, 2)
    return vis


def visualize_compare(img, gt_anns, pred_anns, categories, conf_threshold=0.1):
    """Side-by-side: GT (green) vs predictions (blue=correct, red=wrong, yellow=missed)."""
    vis = img.copy()

    # Filter by confidence
    filtered_preds = [p for p in pred_anns if p.get("score", 0) >= conf_threshold]

    matched, unmatched_preds, unmatched_gts = match_predictions_to_gt(gt_anns, filtered_preds)

    # Draw matched (correct class = blue, wrong class = red)
    correct = 0
    wrong_class = 0
    for gt, pred, iou_val, class_correct in matched:
        if class_correct:
            color = BLUE
            correct += 1
        else:
            color = RED
            wrong_class += 1

        cat_id = pred["category_id"]
        name = categories.get(cat_id, f"id={cat_id}")
        label = f"{shorten_name(name)} {pred['score']:.2f}"
        draw_box(vis, pred["bbox"], color, label)

    # Draw false positives (red, dashed-style thicker)
    for pred in unmatched_preds:
        cat_id = pred["category_id"]
        name = categories.get(cat_id, f"id={cat_id}")
        label = f"FP: {shorten_name(name)} {pred['score']:.2f}"
        draw_box(vis, pred["bbox"], RED, label, thickness=1)

    # Draw missed GT (yellow)
    for gt in unmatched_gts:
        name = categories.get(gt["category_id"], "?")
        label = f"MISS: {shorten_name(name)}"
        draw_box(vis, gt["bbox"], YELLOW, label, thickness=1)

    # Stats
    n_gt = len(gt_anns)
    n_pred = len(filtered_preds)
    n_fp = len(unmatched_preds)
    n_fn = len(unmatched_gts)
    precision = correct / n_pred if n_pred > 0 else 0
    recall = correct / n_gt if n_gt > 0 else 0

    lines = [
        f"GT: {n_gt} | Pred: {n_pred}",
        f"Correct: {correct} | Wrong cls: {wrong_class}",
        f"FP: {n_fp} | Missed: {n_fn}",
        f"P: {precision:.2f} | R: {recall:.2f}",
    ]
    for i, line in enumerate(lines):
        cv2.putText(vis, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 2)
        cv2.putText(vis, line, (10, 25 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)

    return vis


def create_stats_summary(all_gt_anns, all_pred_anns, categories, conf_threshold=0.1):
    """Create aggregate statistics across all images."""
    total_gt = 0
    total_pred = 0
    total_correct = 0
    total_wrong_class = 0
    total_fp = 0
    total_fn = 0
    per_class_correct = {}
    per_class_total = {}

    for img_id in set(list(all_gt_anns.keys()) + list(all_pred_anns.keys())):
        gt = all_gt_anns.get(img_id, [])
        preds = [p for p in all_pred_anns.get(img_id, []) if p.get("score", 0) >= conf_threshold]

        total_gt += len(gt)
        total_pred += len(preds)

        matched, unmatched_preds, unmatched_gts = match_predictions_to_gt(gt, preds)

        for gt_ann, pred, iou_val, class_correct in matched:
            cat_id = gt_ann["category_id"]
            per_class_total[cat_id] = per_class_total.get(cat_id, 0) + 1
            if class_correct:
                total_correct += 1
                per_class_correct[cat_id] = per_class_correct.get(cat_id, 0) + 1
            else:
                total_wrong_class += 1

        total_fp += len(unmatched_preds)
        total_fn += len(unmatched_gts)

        for gt_ann in unmatched_gts:
            cat_id = gt_ann["category_id"]
            per_class_total[cat_id] = per_class_total.get(cat_id, 0) + 1

    precision = total_correct / total_pred if total_pred > 0 else 0
    recall = total_correct / total_gt if total_gt > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    print("\n" + "=" * 60)
    print("AGGREGATE STATISTICS")
    print("=" * 60)
    print(f"Ground truth boxes:  {total_gt}")
    print(f"Prediction boxes:    {total_pred}")
    print(f"Correct (IoU+class): {total_correct}")
    print(f"Wrong class:         {total_wrong_class}")
    print(f"False positives:     {total_fp}")
    print(f"Missed (FN):         {total_fn}")
    print(f"Precision:           {precision:.4f}")
    print(f"Recall:              {recall:.4f}")
    print(f"F1:                  {f1:.4f}")

    # Worst classes (most missed)
    print(f"\n{'WORST CLASSES (most missed)':}")
    print(f"{'Category':<40} {'Correct':>8} {'Total':>8} {'Recall':>8}")
    print("-" * 70)

    class_recalls = []
    for cat_id, total in per_class_total.items():
        correct = per_class_correct.get(cat_id, 0)
        recall = correct / total if total > 0 else 0
        name = categories.get(cat_id, f"id={cat_id}")
        class_recalls.append((name, correct, total, recall))

    class_recalls.sort(key=lambda x: x[3])
    for name, correct, total, recall in class_recalls[:20]:
        print(f"{shorten_name(name, 38):<40} {correct:>8} {total:>8} {recall:>8.2f}")

    print("=" * 60)


def create_montage(images: list[np.ndarray], cols: int = 3, target_h: int = 600) -> np.ndarray:
    """Create a grid montage of images."""
    if not images:
        return np.zeros((100, 100, 3), dtype=np.uint8)

    # Resize all to same height
    resized = []
    for img in images:
        h, w = img.shape[:2]
        scale = target_h / h
        new_w = int(w * scale)
        resized.append(cv2.resize(img, (new_w, target_h)))

    # Pad to same width
    max_w = max(r.shape[1] for r in resized)
    padded = []
    for r in resized:
        if r.shape[1] < max_w:
            pad = np.zeros((target_h, max_w - r.shape[1], 3), dtype=np.uint8)
            r = np.hstack([r, pad])
        padded.append(r)

    # Create grid
    rows_list = []
    for i in range(0, len(padded), cols):
        row_imgs = padded[i:i + cols]
        while len(row_imgs) < cols:
            row_imgs.append(np.zeros_like(padded[0]))
        rows_list.append(np.hstack(row_imgs))

    return np.vstack(rows_list)


def main():
    parser = argparse.ArgumentParser(description="Visualize detections vs ground truth")
    parser.add_argument("--mode", choices=["gt", "pred", "compare"], default="compare")
    parser.add_argument("--images", required=True, help="Directory with images")
    parser.add_argument("--annotations", help="COCO annotations.json")
    parser.add_argument("--predictions", help="predictions.json")
    parser.add_argument("--output", help="Output directory (default: display)")
    parser.add_argument("--image-id", type=int, help="Visualize single image by ID")
    parser.add_argument("--conf", type=float, default=0.1, help="Confidence threshold")
    parser.add_argument("--montage", action="store_true", help="Create grid montage")
    parser.add_argument("--max-images", type=int, default=12, help="Max images in montage")
    parser.add_argument("--stats", action="store_true", help="Print aggregate stats")
    args = parser.parse_args()

    # Load data
    categories = {}
    gt_data = None
    pred_data = None

    if args.annotations:
        gt_data = load_annotations(args.annotations)
        categories = gt_data["categories"]

    if args.predictions:
        pred_data = load_predictions(args.predictions)

    # Find images
    img_dir = Path(args.images)
    image_files = sorted(
        p for p in img_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    if args.image_id is not None:
        image_files = [
            p for p in image_files
            if int(p.stem.split("_")[-1]) == args.image_id
        ]
        if not image_files:
            print(f"Image ID {args.image_id} not found")
            return

    # Output directory
    output_dir = None
    if args.output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

    # Stats
    if args.stats and gt_data and pred_data:
        create_stats_summary(
            gt_data["anns_by_image"], pred_data, categories, args.conf
        )

    # Visualize
    montage_images = []

    for img_path in image_files[:args.max_images if args.montage else len(image_files)]:
        image_id = int(img_path.stem.split("_")[-1])
        img = cv2.imread(str(img_path))
        if img is None:
            continue

        gt_anns = gt_data["anns_by_image"].get(image_id, []) if gt_data else []
        pred_anns = pred_data.get(image_id, []) if pred_data else []

        if args.mode == "gt":
            vis = visualize_gt(img, gt_anns, categories)
        elif args.mode == "pred":
            vis = visualize_pred(img, pred_anns, categories, args.conf)
        else:
            vis = visualize_compare(img, gt_anns, pred_anns, categories, args.conf)

        if args.montage:
            montage_images.append(vis)
        elif output_dir:
            out_path = output_dir / f"viz_{img_path.name}"
            cv2.imwrite(str(out_path), vis)
            print(f"Saved: {out_path}")
        else:
            # Display
            win_name = f"[{args.mode}] {img_path.name} (q=quit, n=next)"
            h, w = vis.shape[:2]
            scale = min(1.0, 1400 / w, 900 / h)
            if scale < 1.0:
                vis_show = cv2.resize(vis, (int(w * scale), int(h * scale)))
            else:
                vis_show = vis
            cv2.imshow(win_name, vis_show)
            key = cv2.waitKey(0) & 0xFF
            cv2.destroyAllWindows()
            if key == ord("q"):
                break

    if args.montage and montage_images:
        montage = create_montage(montage_images, cols=3)
        if output_dir:
            out_path = output_dir / "montage.jpg"
            cv2.imwrite(str(out_path), montage, [cv2.IMWRITE_JPEG_QUALITY, 90])
            print(f"Saved montage: {out_path}")
        else:
            h, w = montage.shape[:2]
            scale = min(1.0, 1800 / w, 1000 / h)
            vis_show = cv2.resize(montage, (int(w * scale), int(h * scale)))
            cv2.imshow("Montage (press any key)", vis_show)
            cv2.waitKey(0)
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
