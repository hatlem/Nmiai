"""NorgesGruppen — Fast & Reliable Submission

Strategy: Speed over complexity. Process ALL images.
  1. Single-class YOLO26 ONNX for detection (best boxes)
  2. Multi-class YOLO26 ONNX for classification (match by IoU)
  3. DINOv2 classifier as tiebreaker when both YOLOs disagree
  4. NO SAHI — just full-image inference at 1280

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

# Try importing modules
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
    from src.utils import enhance_retail_image
    CLAHE_AVAILABLE = True
except ImportError:
    CLAHE_AVAILABLE = False

# ── Config ────────────────────────────────────────────────────────────
TOTAL_TIMEOUT = 285
CONF_THRESHOLD = 0.001
NMS_IOU = 0.65
IMGSZ = 1280
MIN_BOX_SIZE = 5
CROP_PAD = 0.05
CLS_WEIGHT = 0.15  # score = det^1.0 * cls^CLS_WEIGHT


def load_models(model_dir: Path, device: str):
    """Load all available models."""
    single_det = None
    multi_det = None
    classifier = None

    # Single-class detector (primary — best detection boxes)
    for name in ["best.onnx", "best.pt"]:
        p = model_dir / name
        if p.exists() and ONNX_AVAILABLE and name.endswith(".onnx"):
            try:
                single_det = ONNXDetector(str(p), conf_threshold=CONF_THRESHOLD)
                print(f"[LOAD] Single-class detector: {name}")
                break
            except Exception as e:
                print(f"[LOAD] Failed {name}: {e}")
        elif p.exists() and name.endswith(".pt"):
            try:
                from ultralytics import YOLO
                single_det = YOLO(str(p))
                print(f"[LOAD] Single-class detector: {name}")
                break
            except Exception as e:
                print(f"[LOAD] Failed {name}: {e}")

    # Multi-class detector (for category prediction)
    for name in ["multi_best.onnx", "multi_best.pt"]:
        p = model_dir / name
        if p.exists() and ONNX_AVAILABLE and name.endswith(".onnx"):
            try:
                multi_det = ONNXDetector(str(p), conf_threshold=0.05)
                print(f"[LOAD] Multi-class detector: {name}")
                break
            except Exception as e:
                print(f"[LOAD] Failed {name}: {e}")

    # DINOv2 classifier (tiebreaker)
    if CLASSIFIER_AVAILABLE:
        try:
            classifier = ProductClassifier(model_dir / "models", device)
            print(f"[LOAD] Classifier: {classifier.mode}")
        except Exception as e:
            print(f"[LOAD] Classifier failed: {e}")

    return single_det, multi_det, classifier


def match_boxes_iou(det_boxes: np.ndarray, mc_boxes: np.ndarray) -> list:
    """For each detection box, find best matching multi-class box by IoU.

    Returns list of (mc_index, iou) for each det box. mc_index=-1 if no match.
    """
    matches = []
    for i in range(len(det_boxes)):
        best_iou = 0
        best_j = -1
        x1_a, y1_a, x2_a, y2_a = det_boxes[i]
        area_a = (x2_a - x1_a) * (y2_a - y1_a)

        for j in range(len(mc_boxes)):
            x1_b, y1_b, x2_b, y2_b = mc_boxes[j]
            inter_x1 = max(x1_a, x1_b)
            inter_y1 = max(y1_a, y1_b)
            inter_x2 = min(x2_a, x2_b)
            inter_y2 = min(y2_a, y2_b)
            inter = max(0, inter_x2 - inter_x1) * max(0, inter_y2 - inter_y1)
            area_b = (x2_b - x1_b) * (y2_b - y1_b)
            union = area_a + area_b - inter
            iou = inter / max(union, 1e-6)
            if iou > best_iou:
                best_iou = iou
                best_j = j

        matches.append((best_j, best_iou))
    return matches


def process_image(
    img_bgr: np.ndarray,
    single_det,
    multi_det,
    classifier,
    device: str,
) -> list[dict]:
    """Process one image with hybrid strategy."""

    # Step 1: CLAHE
    if CLAHE_AVAILABLE:
        img = enhance_retail_image(img_bgr)
    else:
        img = img_bgr

    # Step 2: Single-class detection (best boxes)
    results = single_det(img, conf=CONF_THRESHOLD, iou=NMS_IOU, imgsz=IMGSZ)
    boxes_obj = results[0].boxes
    if boxes_obj is None or len(boxes_obj) == 0:
        return []

    det_boxes = boxes_obj.xyxy.numpy()
    det_scores = boxes_obj.conf.numpy()

    # Step 3: Multi-class detection for category assignment
    mc_categories = {}  # det_index -> (category_id, confidence)

    if multi_det is not None:
        mc_results = multi_det(img, conf=0.05, iou=0.5, imgsz=IMGSZ)
        mc_boxes_obj = mc_results[0].boxes
        if mc_boxes_obj is not None and len(mc_boxes_obj) > 0:
            mc_boxes = mc_boxes_obj.xyxy.numpy()
            mc_cls = mc_boxes_obj.cls.numpy().astype(int)
            mc_conf = mc_boxes_obj.conf.numpy()

            # Match detection boxes to multi-class boxes
            matches = match_boxes_iou(det_boxes, mc_boxes)
            for i, (mc_idx, iou) in enumerate(matches):
                if mc_idx >= 0 and iou > 0.3:
                    mc_categories[i] = (int(mc_cls[mc_idx]), float(mc_conf[mc_idx]))

    # Step 4: DINOv2 classifier for remaining boxes (or all if no multi-class)
    dinov2_categories = {}

    if classifier is not None and classifier.mode != "none":
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
            cls_results = classifier.classify(crops, batch_size=64)
            for idx, (cat_id, cls_conf) in zip(crop_indices, cls_results):
                dinov2_categories[idx] = (cat_id, cls_conf)

    # Step 5: Merge — pick best category for each detection
    detections = []
    for i in range(len(det_boxes)):
        x1, y1, x2, y2 = det_boxes[i]
        w, h = x2 - x1, y2 - y1
        if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
            continue

        det_score = float(det_scores[i])
        mc_cat = mc_categories.get(i)
        dv_cat = dinov2_categories.get(i)

        # Decision logic:
        if mc_cat and dv_cat:
            mc_id, mc_conf = mc_cat
            dv_id, dv_conf = dv_cat

            if mc_id == dv_id:
                # Both agree — high confidence
                cat_id = mc_id
                cls_conf = max(mc_conf, dv_conf)
            elif mc_conf > 0.3:
                # Multi-class YOLO is confident — trust it (has spatial context)
                cat_id = mc_id
                cls_conf = mc_conf
            else:
                # DINOv2 by default (trained specifically for this)
                cat_id = dv_id
                cls_conf = dv_conf
        elif mc_cat:
            cat_id, cls_conf = mc_cat
        elif dv_cat:
            cat_id, cls_conf = dv_cat
        else:
            cat_id, cls_conf = 0, 0.01

        # Combined score: preserve detection ranking, slight classification boost
        combined = det_score * (max(cls_conf, 1e-6) ** CLS_WEIGHT)

        detections.append({
            "x1": float(x1),
            "y1": float(y1),
            "w": float(w),
            "h": float(h),
            "category_id": int(cat_id),
            "score": combined,
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

    single_det, multi_det, classifier = load_models(model_dir, device)

    if single_det is None:
        print("[ERROR] No single-class detector found!")
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

    # Process all images
    predictions = []
    for img_idx, img_path in enumerate(image_files):
        img_start = time.perf_counter()
        elapsed = time.perf_counter() - t_start
        remaining = TOTAL_TIMEOUT - elapsed

        if remaining < 3:
            print(f"[WARN] {remaining:.0f}s left — stopping at {img_idx}/{num_images}")
            break

        image_id = int(img_path.stem.split("_")[-1])
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue

        # Decide: use classifier only if we have time budget
        time_per_img = elapsed / max(1, img_idx) if img_idx > 0 else 5
        images_left = num_images - img_idx
        use_classifier = classifier if (time_per_img * images_left < remaining * 0.9) else None

        dets = process_image(img_bgr, single_det, multi_det, use_classifier, device)

        for det in dets:
            predictions.append({
                "image_id": int(image_id),
                "category_id": int(det["category_id"]),
                "bbox": [
                    round(det["x1"], 1),
                    round(det["y1"], 1),
                    round(det["w"], 1),
                    round(det["h"], 1),
                ],
                "score": round(det["score"], 4),
            })

        img_time = time.perf_counter() - img_start
        if img_idx < 3 or img_idx % 10 == 0:
            print(
                f"  [{img_idx+1}/{num_images}] {img_path.name}: "
                f"{len(dets)} dets, {img_time:.2f}s | "
                f"remaining: {TOTAL_TIMEOUT - (time.perf_counter() - t_start):.0f}s"
            )

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(predictions, f)

    total = time.perf_counter() - t_start
    print(f"\n[DONE] {len(predictions)} preds for {num_images} imgs in {total:.1f}s")


if __name__ == "__main__":
    main()
