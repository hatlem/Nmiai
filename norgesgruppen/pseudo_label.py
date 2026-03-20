"""Pseudo-labeling pipeline for NorgesGruppen object detection.

Uses the best trained multi-class model to find missed annotations in the
training data. High-confidence predictions that don't overlap with existing
ground truth labels are added as pseudo-labels.

Usage:
    python pseudo_label.py
"""

import torch
_orig_load = torch.load
def _patched_load(f, *a, **kw):
    kw.setdefault("weights_only", False)
    return _orig_load(f, *a, **kw)
torch.load = _patched_load

import shutil
import time
from pathlib import Path

import numpy as np
from ultralytics import YOLO


# --- Config ---
MODEL_PATH = "/tmp/train/runs/detect/train4/weights/best.pt"
DATASET_DIR = Path("/tmp/train/data/yolo_dataset")
PSEUDO_DIR = Path("/tmp/train/data/yolo_pseudo")
CONF_THRESHOLD = 0.5   # Only high-confidence predictions
IOU_THRESHOLD = 0.3    # Max IoU with GT to be considered "new"
IMGSZ = 1280


def compute_iou(box1, box2):
    """Compute IoU between two boxes in xywh format (YOLO normalized)."""
    # Convert xywh to xyxy
    x1_1 = box1[0] - box1[2] / 2
    y1_1 = box1[1] - box1[3] / 2
    x2_1 = box1[0] + box1[2] / 2
    y2_1 = box1[1] + box1[3] / 2

    x1_2 = box2[0] - box2[2] / 2
    y1_2 = box2[1] - box2[3] / 2
    x2_2 = box2[0] + box2[2] / 2
    y2_2 = box2[1] + box2[3] / 2

    inter_x1 = max(x1_1, x1_2)
    inter_y1 = max(y1_1, y1_2)
    inter_x2 = min(x2_1, x2_2)
    inter_y2 = min(y2_1, y2_2)

    inter_area = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
    area1 = box1[2] * box1[3]
    area2 = box2[2] * box2[3]
    union_area = area1 + area2 - inter_area

    if union_area <= 0:
        return 0.0
    return inter_area / union_area


def parse_label_file(label_path):
    """Parse a YOLO label file, return list of (class_id, x, y, w, h)."""
    labels = []
    if not label_path.exists():
        return labels
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                x, y, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
                labels.append((cid, x, y, w, h))
    return labels


def is_novel_prediction(pred_box, gt_labels, iou_thresh):
    """Check if a prediction doesn't overlap with any GT box."""
    for gt in gt_labels:
        gt_box = gt[1:]  # (x, y, w, h)
        if compute_iou(pred_box, gt_box) >= iou_thresh:
            return False
    return True


def process_split(model, split, dataset_dir, pseudo_dir):
    """Run inference on a split and merge predictions with GT labels."""
    img_dir = dataset_dir / "images" / split
    gt_label_dir = dataset_dir / "labels" / split
    pseudo_img_dir = pseudo_dir / "images" / split
    pseudo_label_dir = pseudo_dir / "labels" / split

    pseudo_img_dir.mkdir(parents=True, exist_ok=True)
    pseudo_label_dir.mkdir(parents=True, exist_ok=True)

    image_files = sorted([
        p for p in img_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    ])

    total_gt = 0
    total_pseudo = 0
    total_images = len(image_files)

    print(f"\n{'='*60}")
    print(f"Processing {split}: {total_images} images")
    print(f"{'='*60}")

    # Process in batches for efficiency
    batch_size = 16
    for batch_start in range(0, total_images, batch_size):
        batch_end = min(batch_start + batch_size, total_images)
        batch_files = image_files[batch_start:batch_end]
        batch_paths = [str(p) for p in batch_files]

        # Run inference on batch
        results = model.predict(
            batch_paths,
            imgsz=IMGSZ,
            conf=CONF_THRESHOLD,
            iou=0.5,  # NMS IoU for predictions
            verbose=False,
            device=0,
        )

        for img_path, result in zip(batch_files, results):
            stem = img_path.stem
            gt_label_path = gt_label_dir / f"{stem}.txt"
            gt_labels = parse_label_file(gt_label_path)
            total_gt += len(gt_labels)

            # Get model predictions as normalized xywh
            pseudo_additions = []
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes
                img_h, img_w = result.orig_shape

                for i in range(len(boxes)):
                    cls_id = int(boxes.cls[i].item())
                    conf = float(boxes.conf[i].item())
                    # Get xyxy and convert to normalized xywh
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                    cx = ((x1 + x2) / 2) / img_w
                    cy = ((y1 + y2) / 2) / img_h
                    w = (x2 - x1) / img_w
                    h = (y2 - y1) / img_h
                    pred_box = (cx, cy, w, h)

                    # Only add if it doesn't overlap with any GT
                    if is_novel_prediction(pred_box, gt_labels, IOU_THRESHOLD):
                        pseudo_additions.append((cls_id, cx, cy, w, h, conf))

            total_pseudo += len(pseudo_additions)

            # Write merged label file
            merged_lines = []
            # Keep all GT labels exactly as-is
            for cid, x, y, w, h in gt_labels:
                merged_lines.append(f"{cid} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            # Add pseudo-labels
            for cid, cx, cy, w, h, conf in pseudo_additions:
                merged_lines.append(f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            # Write label
            pseudo_label_path = pseudo_label_dir / f"{stem}.txt"
            pseudo_label_path.write_text("\n".join(merged_lines) + "\n" if merged_lines else "")

            # Symlink image (save disk space)
            pseudo_img_path = pseudo_img_dir / img_path.name
            if not pseudo_img_path.exists():
                pseudo_img_path.symlink_to(img_path.resolve())

        if (batch_end) % 64 == 0 or batch_end == total_images:
            print(f"  Processed {batch_end}/{total_images} images | "
                  f"GT: {total_gt} | Pseudo: {total_pseudo}")

    print(f"\n{split} summary:")
    print(f"  Total GT labels: {total_gt}")
    print(f"  Pseudo-labels added: {total_pseudo}")
    print(f"  Enrichment: +{total_pseudo / max(total_gt, 1) * 100:.1f}%")

    return total_gt, total_pseudo


def create_data_yaml(pseudo_dir, nc=356):
    """Create data.yaml for the pseudo-labeled dataset."""
    # Read original data.yaml to get class names
    orig_yaml = DATASET_DIR / "data.yaml"
    with open(orig_yaml) as f:
        content = f.read()

    # Replace path
    new_content = content.replace(
        f"path: {DATASET_DIR}",
        f"path: {pseudo_dir}"
    )
    # Also handle if path is different format
    if f"path: {pseudo_dir}" not in new_content:
        import re
        new_content = re.sub(r"path:.*", f"path: {pseudo_dir}", new_content)

    yaml_path = pseudo_dir / "data.yaml"
    yaml_path.write_text(new_content)
    print(f"\nCreated {yaml_path}")
    return str(yaml_path)


def main():
    start = time.time()
    print("=" * 60)
    print("PSEUDO-LABELING PIPELINE")
    print(f"Model: {MODEL_PATH}")
    print(f"Confidence threshold: {CONF_THRESHOLD}")
    print(f"IoU threshold: {IOU_THRESHOLD}")
    print(f"Output: {PSEUDO_DIR}")
    print("=" * 60)

    # Clean output directory
    if PSEUDO_DIR.exists():
        print(f"Removing existing {PSEUDO_DIR}")
        shutil.rmtree(PSEUDO_DIR)
    PSEUDO_DIR.mkdir(parents=True)

    # Load model
    print(f"\nLoading model from {MODEL_PATH}...")
    model = YOLO(MODEL_PATH)
    print("Model loaded.")

    # Process both splits
    total_gt = 0
    total_pseudo = 0
    for split in ["train", "val"]:
        gt, pseudo = process_split(model, split, DATASET_DIR, PSEUDO_DIR)
        total_gt += gt
        total_pseudo += pseudo

    # Create data.yaml
    create_data_yaml(PSEUDO_DIR)

    elapsed = time.time() - start
    print(f"\n{'='*60}")
    print(f"DONE in {elapsed:.0f}s")
    print(f"Total GT labels: {total_gt}")
    print(f"Total pseudo-labels added: {total_pseudo}")
    print(f"Total labels in new dataset: {total_gt + total_pseudo}")
    print(f"Enrichment: +{total_pseudo / max(total_gt, 1) * 100:.1f}%")
    print(f"Dataset at: {PSEUDO_DIR}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
