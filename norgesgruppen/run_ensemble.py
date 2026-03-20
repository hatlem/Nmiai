"""NorgesGruppen — 3-Model WBF Ensemble Submission

3 multi-class YOLO26-x models trained on different data:
  1. pseudo_best.onnx — pseudo-labeled data (mAP50=0.789)
  2. fold0_best.onnx  — K-fold 0 (mAP50=0.726)
  3. fold2_best.onnx  — K-fold 2 (mAP50=0.749)

Each model gives BOTH detection AND classification.
WBF fuses predictions from all 3 + optional TTA.

No `import os` — uses pathlib only.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch

try:
    from src.onnx_detector import ONNXDetector
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from src.wbf import weighted_boxes_fusion
    WBF_AVAILABLE = True
except ImportError:
    WBF_AVAILABLE = False

if not WBF_AVAILABLE:
    try:
        from ensemble_boxes import weighted_boxes_fusion
        WBF_AVAILABLE = True
    except ImportError:
        pass

# ── Config ────────────────────────────────────────────────────────────
TOTAL_TIMEOUT = 285
CONF_THRESHOLD = 0.001  # Lower = more recall = better mAP (0.01 gave 0.9139, 0.05 gave 0.9119)
NMS_IOU = 0.65
IMGSZ = 1280
MIN_BOX_SIZE = 4
WBF_IOU_THR = 0.55
WBF_SKIP_THR = 0.001


def load_models(model_dir: Path):
    """Load all available ONNX multi-class detectors."""
    models = []
    names = ["pseudo_best.onnx", "img1600_best.onnx", "fold2_best.onnx",
             "fold0_best.onnx", "fold1_best.onnx", "best.onnx"]

    for name in names:
        p = model_dir / name
        if p.exists() and ONNX_AVAILABLE:
            try:
                m = ONNXDetector(str(p), conf_threshold=CONF_THRESHOLD)
                models.append((name, m))
                print(f"[LOAD] {name}")
            except Exception as e:
                print(f"[LOAD] Failed {name}: {e}")

    return models


def detect_single(model, img_bgr, conf, iou, imgsz):
    """Run single model, return (boxes_xyxy, scores, labels)."""
    results = model(img_bgr, conf=conf, iou=iou, imgsz=imgsz)
    boxes_obj = results[0].boxes
    if boxes_obj is None or len(boxes_obj) == 0:
        return np.zeros((0, 4)), np.array([]), np.array([], dtype=int)
    return (boxes_obj.xyxy.numpy(),
            boxes_obj.conf.numpy(),
            boxes_obj.cls.numpy().astype(int))


def ensemble_detect(models, img_bgr, use_tta=False):
    """Run all models and fuse with WBF."""
    h, w = img_bgr.shape[:2]

    all_boxes, all_scores, all_labels = [], [], []

    for name, model in models:
        # Normal pass at primary resolution
        boxes, scores, labels = detect_single(model, img_bgr, CONF_THRESHOLD, NMS_IOU, IMGSZ)
        if len(boxes) > 0:
            norm_boxes = boxes.copy()
            norm_boxes[:, [0, 2]] /= w
            norm_boxes[:, [1, 3]] /= h
            all_boxes.append(np.clip(norm_boxes, 0, 1))
            all_scores.append(scores)
            all_labels.append(labels)

        if use_tta:
            # TTA 1: horizontal flip at primary resolution
            flipped = cv2.flip(img_bgr, 1)
            fb, fs, fl = detect_single(model, flipped, CONF_THRESHOLD, NMS_IOU, IMGSZ)
            if len(fb) > 0:
                fb_mirror = fb.copy()
                fb_mirror[:, 0] = w - fb[:, 2]
                fb_mirror[:, 2] = w - fb[:, 0]
                fb_mirror[:, [0, 2]] /= w
                fb_mirror[:, [1, 3]] /= h
                all_boxes.append(np.clip(fb_mirror, 0, 1))
                all_scores.append(fs)
                all_labels.append(fl)

    if not all_boxes or not any(len(b) > 0 for b in all_boxes):
        return np.zeros((0, 4)), np.array([]), np.array([], dtype=int)

    # WBF fusion — weight by prediction quality
    # all_boxes has variable length depending on TTA and which models produced detections
    weights = [1.0] * len(all_boxes)
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        weights=weights, iou_thr=WBF_IOU_THR, skip_box_thr=WBF_SKIP_THR,
    )

    fused_labels = np.asarray(fused_labels, dtype=int)
    fused_boxes = np.asarray(fused_boxes)
    fused_scores = np.asarray(fused_scores)

    # Convert back to pixel coords
    if len(fused_boxes) > 0:
        fused_boxes[:, [0, 2]] *= w
        fused_boxes[:, [1, 3]] *= h

    return fused_boxes, fused_scores, fused_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t_start = time.perf_counter()
    model_dir = Path(__file__).parent

    print(f"[INIT] Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")

    models = load_models(model_dir)
    if not models:
        print("[ERROR] No models found!")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump([], f)
        return

    print(f"[INIT] {len(models)} models loaded for ensemble")

    # Discover images
    input_dir = Path(args.input)
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    num_images = len(image_files)
    init_time = time.perf_counter() - t_start
    print(f"[INIT] Setup: {init_time:.1f}s | Images: {num_images}")

    # Estimate timing: run first image to calibrate
    predictions = []
    use_tta = True  # Start optimistic

    for img_idx, img_path in enumerate(image_files):
        elapsed = time.perf_counter() - t_start
        remaining = TOTAL_TIMEOUT - elapsed

        if remaining < 3:
            print(f"[WARN] {remaining:.0f}s left — stopping at {img_idx}/{num_images}")
            break

        image_id = int(img_path.stem.split("_")[-1])
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        img_start = time.perf_counter()

        # After first 2 images, decide if TTA fits in time budget
        if img_idx == 2 and use_tta:
            avg_time = elapsed / max(1, img_idx)
            est_total = avg_time * num_images
            if est_total > TOTAL_TIMEOUT * 0.85:
                use_tta = False
                print(f"[INFO] Disabling TTA — {avg_time:.1f}s/img too slow for {num_images} images")

        # Per-image TTA check
        if use_tta and remaining < (num_images - img_idx) * 3:
            use_tta = False

        boxes, scores, labels = ensemble_detect(models, img_bgr, use_tta=use_tta)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            w, h = x2 - x1, y2 - y1
            if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
                continue
            predictions.append({
                "image_id": int(image_id),
                "category_id": int(labels[i]),
                "bbox": [round(float(x1), 1), round(float(y1), 1),
                         round(float(w), 1), round(float(h), 1)],
                "score": round(float(scores[i]), 4),
            })

        img_time = time.perf_counter() - img_start
        if img_idx < 3 or img_idx % 20 == 0:
            mode = "TTA" if use_tta else "fast"
            print(f"  [{img_idx+1}/{num_images}] {img_path.name}: "
                  f"{sum(1 for p in predictions if p['image_id']==image_id)} dets, "
                  f"{img_time:.2f}s ({mode}) | {remaining:.0f}s left")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(predictions, f)

    total = time.perf_counter() - t_start
    print(f"\n[DONE] {len(predictions)} preds for {num_images} imgs in {total:.1f}s")


if __name__ == "__main__":
    main()
