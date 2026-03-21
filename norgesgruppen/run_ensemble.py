"""NorgesGruppen — 3-Model WBF Ensemble (proven 0.9158 config + YOLO11 diversity)

EXACT config from 0.9158 submission. Only model change: fold2 → YOLO11-x.
No fancy extras — they all hurt score.

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
    from ensemble_boxes import weighted_boxes_fusion
    WBF_AVAILABLE = True
except ImportError:
    WBF_AVAILABLE = False

if not WBF_AVAILABLE:
    try:
        from src.wbf import weighted_boxes_fusion
        WBF_AVAILABLE = True
    except ImportError:
        pass

# ── Config (EXACT match of 0.9158 submission) ────────────────────────
TOTAL_TIMEOUT = 285
CONF_THRESHOLD = 0.001
NMS_IOU = 0.65
IMGSZ = 1280
MIN_BOX_SIZE = 4
WBF_IOU_THR = 0.55
WBF_SKIP_THR = 0.001


def load_models(model_dir: Path):
    """Load all available ONNX multi-class detectors."""
    models = []
    names = ["pseudo_best.onnx", "yolov8x_best.onnx", "yolo11x_best.onnx",
             "img1600_best.onnx", "fold2_best.onnx", "best.onnx"]

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


def ensemble_detect(models, img_bgr, use_tta=True):
    """Run all models with optional flip TTA and fuse with WBF."""
    h, w = img_bgr.shape[:2]
    all_boxes, all_scores, all_labels = [], [], []

    for _name, model in models:
        # Normal pass at 1280
        boxes, scores, labels = detect_single(model, img_bgr, CONF_THRESHOLD, NMS_IOU, IMGSZ)
        if len(boxes) > 0:
            norm = boxes.copy()
            norm[:, [0, 2]] /= w
            norm[:, [1, 3]] /= h
            all_boxes.append(np.clip(norm, 0, 1))
            all_scores.append(scores)
            all_labels.append(labels)

        # Flip TTA
        if use_tta:
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

    # WBF fusion — uniform weights, default avg
    weights = [1.0] * len(all_boxes)
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        weights=weights, iou_thr=WBF_IOU_THR, skip_box_thr=WBF_SKIP_THR,
    )

    fused_labels = np.asarray(fused_labels, dtype=int)
    fused_boxes = np.asarray(fused_boxes)
    fused_scores = np.asarray(fused_scores)

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
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump([], f)
        return

    print(f"[INIT] {len(models)} models loaded")

    input_dir = Path(args.input)
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    num_images = len(image_files)
    print(f"[INIT] {num_images} images, setup: {time.perf_counter()-t_start:.1f}s")

    predictions = []
    use_tta = True

    for img_idx, img_path in enumerate(image_files):
        elapsed = time.perf_counter() - t_start
        remaining = TOTAL_TIMEOUT - elapsed

        if remaining < 3:
            print(f"[WARN] stopping at {img_idx}/{num_images}")
            break

        image_id = int(img_path.stem.split("_")[-1])
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        img_start = time.perf_counter()

        if img_idx == 2 and use_tta:
            avg = (time.perf_counter() - t_start) / max(1, img_idx)
            if avg * num_images > TOTAL_TIMEOUT * 0.85:
                use_tta = False
                print(f"[INFO] TTA disabled ({avg:.1f}s/img)")

        if use_tta and remaining < (num_images - img_idx) * 2:
            use_tta = False

        boxes, scores, labels = ensemble_detect(models, img_bgr, use_tta=use_tta)

        for i in range(len(boxes)):
            x1, y1, x2, y2 = boxes[i]
            bw, bh = x2 - x1, y2 - y1
            if bw < MIN_BOX_SIZE or bh < MIN_BOX_SIZE:
                continue
            predictions.append({
                "image_id": int(image_id),
                "category_id": int(labels[i]),
                "bbox": [round(float(x1), 1), round(float(y1), 1),
                         round(float(bw), 1), round(float(bh), 1)],
                "score": round(float(scores[i]), 4),
            })

        img_time = time.perf_counter() - img_start
        if img_idx < 3 or img_idx % 20 == 0:
            n = sum(1 for p in predictions if p["image_id"] == image_id)
            print(f"  [{img_idx+1}/{num_images}] {n} dets, {img_time:.1f}s | {remaining:.0f}s left")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(predictions, f)

    print(f"\n[DONE] {len(predictions)} preds, {time.perf_counter()-t_start:.1f}s")


if __name__ == "__main__":
    main()
