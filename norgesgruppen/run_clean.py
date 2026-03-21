"""NorgesGruppen — Optimized multi-class submission.

Research-backed inference pipeline:
  1. CLAHE preprocessing
  2. Multi-scale inference (640, 1280) per model — NO augment=True (explicit TTA)
  3. Explicit horizontal flip TTA as separate WBF input
  4. WBF fusion with iou_thr=0.5 (tuned for dense shelves)
  5. NO Soft-NMS after WBF (double-suppression kills recall)
  6. Aspect ratio + area filtering (remove impossible boxes)

No `import os` — uses pathlib only. Sandbox-compatible.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

# Patch torch.load for PyTorch 2.6+ compatibility
_original_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────
TOTAL_TIMEOUT = 280
CONF_THRESHOLD = 0.001
NMS_IOU = 0.65
SCALES = [640, 1280]       # 2 scales per pass + explicit flip = 4 WBF inputs
MIN_BOX_SIZE = 4
MAX_ASPECT_RATIO = 6.0     # Filter boxes with ratio > 6:1
MIN_AREA_RATIO = 0.0001    # Filter boxes smaller than 0.01% of image

# WBF
try:
    from ensemble_boxes import weighted_boxes_fusion
    WBF_LIB = True
except ImportError:
    WBF_LIB = False

if not WBF_LIB:
    from src.wbf import weighted_boxes_fusion


def enhance(img):
    """CLAHE in LAB color space."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return cv2.cvtColor(cv2.merge([clahe.apply(l), a, b]), cv2.COLOR_LAB2BGR)


def run_model(model, img, device, scale):
    """Single inference pass, returns normalized boxes."""
    h, w = img.shape[:2]
    results = model(img, device=device, verbose=False,
                    conf=CONF_THRESHOLD, iou=NMS_IOU, imgsz=scale, augment=False)
    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            return np.zeros((0, 4)), np.array([]), np.array([])
        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        labels = r.boxes.cls.cpu().numpy().astype(int)
        boxes[:, [0, 2]] /= w
        boxes[:, [1, 3]] /= h
        return np.clip(boxes, 0, 1), scores, labels
    return np.zeros((0, 4)), np.array([]), np.array([])


def detect_with_tta(model, img, device):
    """Multi-scale + explicit flip TTA, fused with WBF."""
    h, w = img.shape[:2]
    all_boxes, all_scores, all_labels = [], [], []
    weights = []

    # Multi-scale passes (original)
    for scale in SCALES:
        boxes, scores, labels = run_model(model, img, device, scale)
        all_boxes.append(boxes if len(boxes) > 0 else np.zeros((0, 4)))
        all_scores.append(scores if len(scores) > 0 else np.array([]))
        all_labels.append(labels if len(labels) > 0 else np.array([]))
        weights.append(1.0 if scale == 1280 else 0.8)  # Higher weight for larger scale

    # Explicit horizontal flip TTA at largest scale
    img_flip = cv2.flip(img, 1)
    boxes_f, scores_f, labels_f = run_model(model, img_flip, device, SCALES[-1])
    if len(boxes_f) > 0:
        # Mirror boxes back: x_new = 1 - x_old (normalized coords)
        boxes_f_mirror = boxes_f.copy()
        boxes_f_mirror[:, 0] = 1.0 - boxes_f[:, 2]  # new x1 = 1 - old x2
        boxes_f_mirror[:, 2] = 1.0 - boxes_f[:, 0]  # new x2 = 1 - old x1
        all_boxes.append(np.clip(boxes_f_mirror, 0, 1))
        all_scores.append(scores_f)
        all_labels.append(labels_f)
        weights.append(0.9)  # Slightly lower weight for flipped
    else:
        all_boxes.append(np.zeros((0, 4)))
        all_scores.append(np.array([]))
        all_labels.append(np.array([]))
        weights.append(0.9)

    if not any(len(b) > 0 for b in all_boxes):
        return np.zeros((0, 4)), np.array([]), np.array([], dtype=int)

    # WBF fusion — iou_thr=0.5 for dense shelves
    fb, fs, fl = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        weights=weights, iou_thr=0.5, skip_box_thr=0.001)

    fl = np.asarray(fl, dtype=int)
    fb = np.asarray(fb)
    fs = np.asarray(fs)

    if len(fb) > 0:
        fb[:, [0, 2]] *= w
        fb[:, [1, 3]] *= h

    return fb, fs, fl


def filter_boxes(boxes, scores, labels, img_h, img_w):
    """Remove impossible boxes: extreme aspect ratio or tiny area."""
    if len(boxes) == 0:
        return boxes, scores, labels

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]

    # Aspect ratio filter
    ratios = np.maximum(widths, heights) / np.maximum(np.minimum(widths, heights), 1e-6)
    ratio_ok = ratios <= MAX_ASPECT_RATIO

    # Area filter
    areas = widths * heights
    img_area = img_h * img_w
    area_ok = areas >= (img_area * MIN_AREA_RATIO)

    # Size filter
    size_ok = (widths >= MIN_BOX_SIZE) & (heights >= MIN_BOX_SIZE)

    keep = ratio_ok & area_ok & size_ok
    return boxes[keep], scores[keep], labels[keep]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = Path(__file__).parent

    model = YOLO(str(model_dir / "best.pt"))
    print(f"[INIT] Device: {device}, Model loaded")

    images = sorted(p for p in Path(args.input).iterdir()
                    if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    print(f"[INIT] {len(images)} images, setup: {time.perf_counter()-t0:.1f}s")

    predictions = []
    for idx, img_path in enumerate(images):
        if TOTAL_TIMEOUT - (time.perf_counter() - t0) < 5:
            print(f"[WARN] Timeout — processed {idx}/{len(images)}")
            break

        img = enhance(cv2.imread(str(img_path)))
        image_id = int(img_path.stem.split("_")[-1])
        img_h, img_w = img.shape[:2]

        boxes, scores, labels = detect_with_tta(model, img, device)

        # Filter impossible boxes (NO Soft-NMS — WBF already handles suppression)
        boxes, scores, labels = filter_boxes(boxes, scores, labels, img_h, img_w)

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            predictions.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [round(float(x1), 1), round(float(y1), 1),
                         round(float(w), 1), round(float(h), 1)],
                "score": round(float(score), 4),
            })

        if idx < 3 or idx % 20 == 0:
            n = sum(1 for p in predictions if p["image_id"] == image_id)
            t = time.perf_counter() - t0
            print(f"  [{idx+1}/{len(images)}] {n} dets | {TOTAL_TIMEOUT-t:.0f}s left")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(predictions, f)

    print(f"\n[DONE] {len(predictions)} predictions, {time.perf_counter()-t0:.1f}s")


if __name__ == "__main__":
    main()
