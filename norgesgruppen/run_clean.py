"""NorgesGruppen — Clean multi-class YOLOv8x submission.

Single model does BOTH detection AND classification. No two-stage pipeline.
Score = detection confidence (includes class probability from YOLO head).

Pipeline:
  1. CLAHE preprocessing
  2. Multi-scale inference (640 + 1280) with TTA
  3. WBF fusion
  4. Output predictions with YOLO's category_id directly

No `import os` — uses pathlib only. Sandbox-compatible.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

# ── Config ────────────────────────────────────────────────────────────
TOTAL_TIMEOUT = 280
CONF_THRESHOLD = 0.001
NMS_IOU = 0.65
SCALES = [640, 1280]
MIN_BOX_SIZE = 4

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


def detect_multiscale(model, img, device):
    """Run at multiple scales, fuse with WBF."""
    h, w = img.shape[:2]
    all_boxes, all_scores, all_labels = [], [], []

    for scale in SCALES:
        results = model(img, device=device, verbose=False,
                        conf=CONF_THRESHOLD, iou=NMS_IOU, imgsz=scale, augment=True)
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                all_boxes.append(np.zeros((0, 4)))
                all_scores.append(np.array([]))
                all_labels.append(np.array([]))
                continue
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            labels = r.boxes.cls.cpu().numpy().astype(int)
            boxes[:, [0, 2]] /= w
            boxes[:, [1, 3]] /= h
            all_boxes.append(np.clip(boxes, 0, 1))
            all_scores.append(scores)
            all_labels.append(labels)

    if not any(len(b) > 0 for b in all_boxes):
        return np.zeros((0, 4)), np.array([]), np.array([], dtype=int)

    fb, fs, fl = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        weights=[1.0] * len(all_boxes), iou_thr=0.55, skip_box_thr=0.001)

    fl = np.asarray(fl, dtype=int)
    fb = np.asarray(fb)
    fs = np.asarray(fs)

    if len(fb) > 0:
        fb[:, [0, 2]] *= w
        fb[:, [1, 3]] *= h

    return fb, fs, fl


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

        boxes, scores, labels = detect_multiscale(model, img, device)

        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
                continue
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
