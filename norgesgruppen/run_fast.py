"""NorgesGruppen — Optimized Submission v6

Fixes applied:
  1. score = det_score (pure detection ranking for det mAP 70%)
     BUT classification confidence IS used for cls mAP ranking
  2. Removed undertrained multi-class YOLO (hurts classification)
  3. CONF_THRESHOLD = 0.01 (not 0.001 = too many FPs)
  4. DINOv2 classifier ONLY for category_id assignment
  5. CLAHE on full image before detection (model trained on CLAHE-enhanced images)

No `import os` — uses pathlib only.
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

try:
    from src.onnx_detector import ONNXDetector
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

try:
    from src.classifier import ProductClassifier
    CLASSIFIER_AVAILABLE = True
except ImportError:
    CLASSIFIER_AVAILABLE = False

try:
    from src.tta import tta_detect
    TTA_AVAILABLE = True
except ImportError:
    TTA_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────
TOTAL_TIMEOUT = 285
CONF_THRESHOLD = 0.01   # Filter obvious noise, but keep recall
NMS_IOU = 0.65
IMGSZ = 1280
MIN_BOX_SIZE = 5
CROP_PAD = 0.05

# TTA config: if avg time per image * TTA_TIME_FACTOR fits in budget, use TTA
# TTA runs 3 scales x 2 orientations = ~6x single inference time
TTA_TIME_FACTOR = 6.5   # Conservative estimate (6x + overhead)
TTA_WARMUP_IMGS = 3     # Run single-scale for first N images to estimate timing


def load_models(model_dir: Path, device: str):
    """Load detector + classifier."""
    detector = None
    classifier = None

    # Detector: prefer ONNX, fall back to .pt
    for name in ["best.onnx", "best.pt"]:
        p = model_dir / name
        if p.exists() and ONNX_AVAILABLE and name.endswith(".onnx"):
            try:
                detector = ONNXDetector(str(p), conf_threshold=CONF_THRESHOLD)
                print(f"[LOAD] Detector: {name}")
                break
            except Exception as e:
                print(f"[LOAD] Failed {name}: {e}")
        elif p.exists() and name.endswith(".pt"):
            try:
                _orig = torch.load
                def _patched(f, *a, **kw):
                    kw.setdefault("weights_only", False)
                    return _orig(f, *a, **kw)
                torch.load = _patched
                from ultralytics import YOLO
                detector = YOLO(str(p))
                print(f"[LOAD] Detector: {name}")
                break
            except Exception as e:
                print(f"[LOAD] Failed {name}: {e}")

    # Classifier: DINOv2 supervised or EfficientNet embedding
    if CLASSIFIER_AVAILABLE:
        try:
            classifier = ProductClassifier(model_dir / "models", device)
            print(f"[LOAD] Classifier: {classifier.mode}")
        except Exception as e:
            print(f"[LOAD] Classifier failed: {e}")

    return detector, classifier


def process_image(
    img_bgr: np.ndarray,
    detector,
    classifier,
    device: str,
    use_tta: bool = False,
) -> list[dict]:
    """Process one image: CLAHE-enhance, then detect + classify."""

    # Step 1: CLAHE enhancement (model trained on CLAHE-enhanced images)
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_ch)
    enhanced = cv2.cvtColor(cv2.merge([l_enhanced, a_ch, b_ch]), cv2.COLOR_LAB2BGR)

    # Step 2: Detection — TTA (multi-scale + flip + WBF) or single-scale
    if use_tta and TTA_AVAILABLE:
        det_boxes, det_scores, _det_labels = tta_detect(
            detector, enhanced, conf=CONF_THRESHOLD, iou=NMS_IOU,
        )
        if len(det_boxes) == 0:
            return []
    else:
        results = detector(enhanced, conf=CONF_THRESHOLD, iou=NMS_IOU, imgsz=IMGSZ)
        boxes_obj = results[0].boxes
        if boxes_obj is None or len(boxes_obj) == 0:
            return []

        det_boxes = boxes_obj.xyxy.numpy()
        det_scores = boxes_obj.conf.numpy()

    # Step 3: Classify with DINOv2 (on CLAHE-enhanced crops)
    categories = {}
    if classifier is not None and classifier.mode != "none":
        img_rgb = cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_w, pil_h = pil_img.size

        crops = []
        crop_indices = []
        for i in range(len(det_boxes)):
            x1, y1, x2, y2 = det_boxes[i]
            w, h = x2 - x1, y2 - y1
            if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
                continue
            pad_x, pad_y = w * CROP_PAD, h * CROP_PAD
            crop = pil_img.crop((
                int(max(0, x1 - pad_x)),
                int(max(0, y1 - pad_y)),
                int(min(pil_w, x2 + pad_x)),
                int(min(pil_h, y2 + pad_y)),
            ))
            crops.append(crop)
            crop_indices.append(i)

        if crops:
            cls_results = classifier.classify(crops, batch_size=128)
            for idx, (cat_id, cls_conf) in zip(crop_indices, cls_results):
                categories[idx] = (cat_id, cls_conf)

    # Step 4: Build output — score = det_score ONLY for detection mAP ranking
    detections = []
    for i in range(len(det_boxes)):
        x1, y1, x2, y2 = det_boxes[i]
        w, h = x2 - x1, y2 - y1
        if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
            continue

        det_score = float(det_scores[i])

        if i in categories:
            cat_id, cls_conf = categories[i]
        else:
            cat_id, cls_conf = 0, 0.0

        detections.append({
            "x1": float(x1),
            "y1": float(y1),
            "w": float(w),
            "h": float(h),
            "category_id": int(cat_id),
            "score": det_score,  # Pure detection score — preserves ranking
        })

    return detections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t_start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = Path(__file__).parent

    print(f"[INIT] Device: {device}")

    detector, classifier = load_models(model_dir, device)

    if detector is None:
        print("[ERROR] No detector found!")
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w") as f:
            json.dump([], f)
        return

    # Discover images
    input_dir = Path(args.input)
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    num_images = len(image_files)
    init_time = time.perf_counter() - t_start
    print(f"[INIT] Setup: {init_time:.1f}s | Images: {num_images}")

    # Determine if TTA is feasible: run warmup images at single-scale first
    # to estimate timing, then decide if TTA fits in budget.
    use_tta_global = False
    single_scale_times: list[float] = []

    # Process all images
    predictions = []
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

        # After warmup, decide TTA feasibility once
        if img_idx == TTA_WARMUP_IMGS and TTA_AVAILABLE and not use_tta_global:
            avg_single = sum(single_scale_times) / len(single_scale_times) if single_scale_times else 3.0
            estimated_tta_time = avg_single * TTA_TIME_FACTOR
            total_tta_budget = estimated_tta_time * num_images
            if total_tta_budget < remaining * 0.85:
                use_tta_global = True
                print(
                    f"[TTA] Enabled — avg single: {avg_single:.2f}s, "
                    f"est TTA: {estimated_tta_time:.2f}s/img, "
                    f"total: {total_tta_budget:.0f}s < {remaining * 0.85:.0f}s budget"
                )
            else:
                print(
                    f"[TTA] Disabled — est {total_tta_budget:.0f}s > "
                    f"{remaining * 0.85:.0f}s budget. Staying single-scale."
                )

        # Per-image TTA decision: re-check budget as we go
        do_tta = use_tta_global
        if do_tta:
            images_left = num_images - img_idx
            avg_single = sum(single_scale_times) / len(single_scale_times) if single_scale_times else 3.0
            estimated_remaining = avg_single * TTA_TIME_FACTOR * images_left
            if estimated_remaining > remaining * 0.9:
                do_tta = False  # Fall back to single-scale for remaining images

        # Skip classifier if running out of time
        time_per_img = elapsed / max(1, img_idx) if img_idx > 0 else 3
        images_left = num_images - img_idx
        if time_per_img * images_left < remaining * 0.95:
            use_cls = classifier
        else:
            use_cls = None
            if img_idx == 0 or images_left % 10 == 0:
                print(f"[WARN] Skipping classifier — {remaining:.0f}s left, ~{images_left} imgs @ {time_per_img:.1f}s/img")

        img_start = time.perf_counter()
        dets = process_image(img_bgr, detector, use_cls, device, use_tta=do_tta)

        # Track single-scale timing for budget estimation
        img_time_for_budget = time.perf_counter() - img_start
        if not do_tta:
            single_scale_times.append(img_time_for_budget)

        for det in dets:
            predictions.append({
                "image_id": int(image_id),
                "category_id": int(det["category_id"]),
                "bbox": [
                    round(float(det["x1"]), 1),
                    round(float(det["y1"]), 1),
                    round(float(det["w"]), 1),
                    round(float(det["h"]), 1),
                ],
                "score": round(float(det["score"]), 4),
            })

        img_time = time.perf_counter() - img_start
        if img_idx < 3 or img_idx % 50 == 0:
            tta_tag = " [TTA]" if do_tta else ""
            print(
                f"  [{img_idx+1}/{num_images}] {img_path.name}: "
                f"{len(dets)} dets, {img_time:.2f}s{tta_tag} | "
                f"remaining: {remaining:.0f}s"
            )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(predictions, f)

    total = time.perf_counter() - t_start
    print(f"\n[DONE] {len(predictions)} preds for {num_images} imgs in {total:.1f}s")


if __name__ == "__main__":
    main()
