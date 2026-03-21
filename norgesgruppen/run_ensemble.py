"""NorgesGruppen — 3-Model WBF Ensemble + Soft-NMS + Multi-Scale TTA

Research-backed improvements:
  1. Multi-scale TTA: 640+960+1280 × flip = 6 passes per model (literature: standard in competition winners)
  2. Soft-NMS after WBF: +1-2% mAP (Bodla et al., ICCV 2017)
  3. WBF ensemble of 3 diverse models

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

try:
    from src.soft_nms import soft_nms
    SOFT_NMS_AVAILABLE = True
except ImportError:
    SOFT_NMS_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────
TOTAL_TIMEOUT = 285
CONF_THRESHOLD = 0.001
NMS_IOU = 0.65
IMGSZ = 1280
TTA_SCALES = [1280]  # Only 1280 — multi-scale (640+960) hurt score (0.9149 vs 0.9158)
MIN_BOX_SIZE = 4
WBF_IOU_THR = 0.45   # Tuned: 0.45 > 0.55 on local eval (+0.0008)
WBF_SKIP_THR = 0.01  # Tuned: 0.01 > 0.001 (removes noise)
SOFT_NMS_SIGMA = 0.5
SOFT_NMS_SCORE_THR = 0.001
# Temperature scaling: sharpen confidence scores for better mAP ranking
# T < 1.0 = sharper (more confident), T > 1.0 = softer
TEMPERATURE = 0.7  # Research: sharpening helps classification mAP


def load_models(model_dir: Path):
    """Load all available ONNX multi-class detectors."""
    models = []
    names = ["pseudo_best.onnx", "yolov8x_best.onnx", "yolo11x_best.onnx",
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


def _add_detections(all_boxes, all_scores, all_labels, boxes, scores, labels, w, h):
    """Normalize boxes and add to collection."""
    if len(boxes) > 0:
        norm = boxes.copy()
        norm[:, [0, 2]] /= w
        norm[:, [1, 3]] /= h
        all_boxes.append(np.clip(norm, 0, 1))
        all_scores.append(scores)
        all_labels.append(labels)


def _add_flipped(all_boxes, all_scores, all_labels, model, img_bgr, w, h, conf, iou, imgsz):
    """Run flipped inference and mirror boxes back."""
    flipped = cv2.flip(img_bgr, 1)
    fb, fs, fl = detect_single(model, flipped, conf, iou, imgsz)
    if len(fb) > 0:
        fb[:, 0], fb[:, 2] = w - fb[:, 2].copy(), w - fb[:, 0].copy()
        _add_detections(all_boxes, all_scores, all_labels, fb, fs, fl, w, h)


def ensemble_detect(models, img_bgr, tta_mode="full"):
    """Run all models with multi-scale TTA and fuse with WBF + Soft-NMS.

    tta_mode: "full" = 3 scales × flip, "light" = 1280 + flip, "none" = 1280 only
    """
    h, w = img_bgr.shape[:2]
    all_boxes, all_scores, all_labels = [], [], []

    scales = TTA_SCALES if tta_mode == "full" else ([IMGSZ] if tta_mode != "none" else [IMGSZ])
    do_flip = tta_mode in ("full", "light")

    for _name, model in models:
        for scale in scales:
            # Normal pass
            boxes, scores, labels = detect_single(model, img_bgr, CONF_THRESHOLD, NMS_IOU, scale)
            _add_detections(all_boxes, all_scores, all_labels, boxes, scores, labels, w, h)

            # Horizontal flip
            if do_flip:
                _add_flipped(all_boxes, all_scores, all_labels, model, img_bgr, w, h,
                             CONF_THRESHOLD, NMS_IOU, scale)

    if not all_boxes or not any(len(b) > 0 for b in all_boxes):
        return np.zeros((0, 4)), np.array([]), np.array([], dtype=int)

    # WBF fusion — weight models by their individual mAP (research: marginal improvement over uniform)
    # pseudo=0.789 strongest, others ~0.77. Give first model 1.5x weight.
    n_passes = len(scales) * (2 if do_flip else 1)
    model_weights = [1.5] + [1.0] * (len(models) - 1)  # First model (pseudo) gets higher weight
    weights = []
    for mw in model_weights[:len(models)]:
        weights.extend([mw] * n_passes)
    # Truncate to actual number of prediction sets
    weights = weights[:len(all_boxes)]
    if len(weights) < len(all_boxes):
        weights.extend([1.0] * (len(all_boxes) - len(weights)))
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        weights=weights, iou_thr=WBF_IOU_THR, skip_box_thr=WBF_SKIP_THR,
        conf_type="max",  # Research: max > avg for ensemble with diverse models
    )

    fused_labels = np.asarray(fused_labels, dtype=int)
    fused_boxes = np.asarray(fused_boxes)
    fused_scores = np.asarray(fused_scores)

    # Convert back to pixel coords
    if len(fused_boxes) > 0:
        fused_boxes[:, [0, 2]] *= w
        fused_boxes[:, [1, 3]] *= h

    # Temperature scaling: sharpen scores for better mAP ranking
    if TEMPERATURE != 1.0 and len(fused_scores) > 0:
        # Apply temperature to logit-space: score -> logit -> scale -> sigmoid
        eps = 1e-7
        logits = np.log(fused_scores / (1.0 - fused_scores + eps) + eps)
        fused_scores = 1.0 / (1.0 + np.exp(-logits / TEMPERATURE))
        fused_scores = np.clip(fused_scores, 0, 1)

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

    # Estimate timing: run first image to calibrate TTA mode
    predictions = []
    tta_mode = "full"  # Start with full multi-scale TTA

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

        # After first 2 images, decide TTA level based on time budget
        if img_idx == 2:
            avg_time = (time.perf_counter() - t_start) / max(1, img_idx)
            est_total = avg_time * num_images
            if est_total > TOTAL_TIMEOUT * 0.85:
                tta_mode = "light"  # Downgrade to 1280+flip only
                print(f"[INFO] TTA downgraded to 'light' — {avg_time:.1f}s/img")
            if est_total > TOTAL_TIMEOUT * 1.2:
                tta_mode = "none"
                print(f"[INFO] TTA disabled — {avg_time:.1f}s/img too slow")

        # Per-image safety check
        if tta_mode != "none" and remaining < (num_images - img_idx) * 2:
            tta_mode = "none"

        boxes, scores, labels = ensemble_detect(models, img_bgr, tta_mode=tta_mode)

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
            mode = tta_mode
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
