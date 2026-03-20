"""NorgesGruppen Object Detection — Best Combined Submission

Integrates all improvements:
  1. CLAHE preprocessing (glare removal)
  2. SAHI tiled inference (small object detection)
  3. Multi-model ensemble (if multiple models exist)
  4. Soft-NMS (dense shelf-aware suppression)
  5. Two-stage classification (DINOv2/EfficientNet-B3 auto-selection)
  6. Time budget manager (skips SAHI if running slow)

Graceful fallbacks:
  - Only best.pt -> single model mode
  - best.pt + rtdetr_best.pt -> ensemble mode
  - No classifier weights -> detection-only (category_id=0, 70% of score)
  - SAHI unavailable -> simple multi-scale inference

No `import os` — uses pathlib only. Sandbox-compatible.

Usage (by sandbox):
    python run_best.py --input /data/images --output /output/predictions.json
"""

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

# Patch torch.load for PyTorch 2.6+ / ultralytics 8.1.0 compatibility
# PyTorch 2.6 defaults weights_only=True, but ultralytics 8.1.0 doesn't pass it
_original_torch_load = torch.load
def _patched_torch_load(f, *args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _original_torch_load(f, *args, **kwargs)
torch.load = _patched_torch_load

from ultralytics import YOLO

from src.classifier import ProductClassifier
from src.utils import enhance_retail_image
from src.wbf import weighted_boxes_fusion

# Try importing ONNX detector
try:
    from src.onnx_detector import ONNXDetector
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

# Try importing optional modules — fall back gracefully
try:
    from src.sahi import sahi_inference
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False

try:
    from src.soft_nms import soft_nms, class_agnostic_soft_nms
    SOFT_NMS_AVAILABLE = True
except ImportError:
    SOFT_NMS_AVAILABLE = False

try:
    from src.ensemble import ensemble_inference
    ENSEMBLE_AVAILABLE = True
except ImportError:
    ENSEMBLE_AVAILABLE = False

# ── Configuration ──────────────────────────────────────────────────────
TOTAL_TIMEOUT = 280          # seconds — leave 20s margin from 300s limit
SAHI_TIME_BUDGET_RATIO = 0.7 # if avg time > budget, skip SAHI for rest

# Detection
CONF_THRESHOLD = 0.001
NMS_IOU = 0.65
IMGSZ_FULL = 1280
IMGSZ_SAHI_TILE = 640
SAHI_OVERLAP = 0.3

# Multi-scale WBF (fallback when SAHI unavailable)
SCALES = [640, 1280]
WBF_IOU_THR = 0.6
WBF_SKIP_BOX_THR = 0.001

# Soft-NMS
SOFT_NMS_SIGMA = 0.5
SOFT_NMS_SCORE_THR = 0.01

# Classification
CLASSIFIER_BATCH_SIZE = 64
CROP_PAD_RATIO = 0.05
MIN_BOX_SIZE = 5


# ── Time Budget Manager ───────────────────────────────────────────────

class TimeBudget:
    """Track elapsed time and decide whether to use SAHI or fast mode."""

    def __init__(self, total_budget: float, num_images: int):
        self.total_budget = total_budget
        self.num_images = max(1, num_images)
        self.start_time = time.perf_counter()
        self.image_times: list[float] = []
        self.sahi_used = 0
        self.sahi_skipped = 0

    def elapsed(self) -> float:
        return time.perf_counter() - self.start_time

    def remaining(self) -> float:
        return max(0.0, self.total_budget - self.elapsed())

    def record_image(self, duration: float) -> None:
        self.image_times.append(duration)

    def should_use_sahi(self, images_remaining: int) -> bool:
        """Decide if we can afford SAHI on the next image."""
        if not SAHI_AVAILABLE:
            return False

        remaining_time = self.remaining()
        if remaining_time < 10:
            # Less than 10s left — always skip SAHI
            return False

        if len(self.image_times) < 2:
            # Not enough data yet — try SAHI for first images
            return True

        avg_time = np.mean(self.image_times)
        time_per_image_budget = remaining_time / max(1, images_remaining)

        # If average time is more than 70% of per-image budget, skip SAHI
        if avg_time > time_per_image_budget * SAHI_TIME_BUDGET_RATIO:
            return False

        return True

    def summary(self) -> str:
        avg = np.mean(self.image_times) if self.image_times else 0
        return (
            f"Time: {self.elapsed():.1f}s total, {avg:.2f}s/img avg | "
            f"SAHI used: {self.sahi_used}, skipped: {self.sahi_skipped}"
        )


# ── Detection Strategies ───────────────────────────────────────────────

def detect_simple(model, img: np.ndarray, device: str):
    """Simple single-pass detection at IMGSZ_FULL with TTA."""
    results = model(
        img,
        device=device,
        verbose=False,
        conf=CONF_THRESHOLD,
        iou=NMS_IOU,
        imgsz=IMGSZ_FULL,
        augment=False,
    )

    boxes_list, scores_list = [], []
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            boxes_list.append(r.boxes.xyxy.cpu().numpy())
            scores_list.append(r.boxes.conf.cpu().numpy())

    if not boxes_list:
        return np.zeros((0, 4)), np.array([])

    return np.concatenate(boxes_list), np.concatenate(scores_list)


def detect_multiscale_wbf(model, img: np.ndarray, device: str):
    """Multi-scale inference with WBF fusion — fallback when SAHI unavailable."""
    img_h, img_w = img.shape[:2]

    all_boxes, all_scores, all_labels = [], [], []

    for scale in SCALES:
        results = model(
            img,
            device=device,
            verbose=False,
            conf=CONF_THRESHOLD,
            iou=NMS_IOU,
            imgsz=scale,
            augment=False,
        )

        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                all_boxes.append(np.zeros((0, 4)))
                all_scores.append(np.array([]))
                all_labels.append(np.array([]))
                continue

            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            labels = r.boxes.cls.cpu().numpy().astype(int)

            boxes[:, [0, 2]] /= img_w
            boxes[:, [1, 3]] /= img_h
            boxes = np.clip(boxes, 0, 1)

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

    if not any(len(b) > 0 for b in all_boxes):
        return np.zeros((0, 4)), np.array([]), np.array([], dtype=int)

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        all_boxes, all_scores, all_labels,
        weights=[1.0] * len(SCALES),
        iou_thr=WBF_IOU_THR,
        skip_box_thr=WBF_SKIP_BOX_THR,
    )

    if len(fused_boxes) > 0:
        fused_boxes[:, [0, 2]] *= img_w
        fused_boxes[:, [1, 3]] *= img_h

    return fused_boxes, fused_scores, fused_labels


def detect_sahi(model, img: np.ndarray, device: str):
    """SAHI tiled inference — best for small objects."""
    return sahi_inference(
        model, img, device,
        slice_size=IMGSZ_SAHI_TILE,
        overlap_ratio=SAHI_OVERLAP,
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=NMS_IOU,
        full_image_scale=IMGSZ_FULL,
    )


def detect_ensemble(models: list, img: np.ndarray, device: str):
    """Multi-model ensemble inference."""
    return ensemble_inference(
        models, img, device,
        scales=[IMGSZ_FULL] * len(models),
        conf_threshold=CONF_THRESHOLD,
        iou_threshold=NMS_IOU,
    )


# ── Main Pipeline ─────────────────────────────────────────────────────

def process_image(
    models: list,
    img_bgr: np.ndarray,
    device: str,
    use_sahi: bool,
    classifier: ProductClassifier,
) -> list[dict]:
    """Full pipeline for a single image.

    1. CLAHE enhance
    2. Detect (SAHI/ensemble/multi-scale/simple — best available)
    3. Soft-NMS on merged detections
    4. Classify crops with best available classifier (DINOv2/EfficientNet)
    5. Combine detection score x classification confidence

    Returns list of (box_xyxy, w, h, category_id, score) dicts ready for output.
    """
    # Step 1: CLAHE enhancement
    img = enhance_retail_image(img_bgr)
    img_h, img_w = img.shape[:2]

    # Step 2: Detection — pick best available strategy
    primary_model = models[0]
    has_multi_model = len(models) > 1

    if use_sahi and SAHI_AVAILABLE:
        # SAHI provides the best small-object recall
        boxes, scores, labels = detect_sahi(primary_model, img, device)
    elif has_multi_model and ENSEMBLE_AVAILABLE:
        boxes, scores, labels = detect_ensemble(models, img, device)
    else:
        # Fast mode: single-pass detection at full resolution
        results = primary_model(
            img,
            device=device,
            verbose=False,
            conf=CONF_THRESHOLD,
            iou=NMS_IOU,
            imgsz=IMGSZ_FULL,
            augment=False,
        )
        all_b, all_s, all_l = [], [], []
        for r in results:
            if r.boxes is not None and len(r.boxes) > 0:
                all_b.append(r.boxes.xyxy.cpu().numpy())
                all_s.append(r.boxes.conf.cpu().numpy())
                all_l.append(r.boxes.cls.cpu().numpy().astype(int))
        if all_b:
            boxes = np.concatenate(all_b)
            scores = np.concatenate(all_s)
            labels = np.concatenate(all_l)
        else:
            boxes = np.zeros((0, 4))
            scores = np.array([])
            labels = np.array([], dtype=int)

    if len(boxes) == 0:
        return []

    # If we have multiple models and used SAHI on primary, also run simple
    # detection on secondary models and merge
    if has_multi_model and not ENSEMBLE_AVAILABLE and len(models) > 1:
        for extra_model in models[1:]:
            extra_boxes, extra_scores = detect_simple(extra_model, img, device)
            if len(extra_boxes) > 0:
                # labels from simple detection are class 0 (single-class detector)
                extra_labels = np.zeros(len(extra_boxes), dtype=int)
                boxes = np.concatenate([boxes, extra_boxes])
                scores = np.concatenate([scores, extra_scores])
                labels = np.concatenate([labels, extra_labels])

    # Step 3: Soft-NMS — preserve overlapping products on dense shelves
    if SOFT_NMS_AVAILABLE and len(boxes) > 0:
        boxes, scores, labels = soft_nms(
            boxes, scores, labels,
            sigma=SOFT_NMS_SIGMA,
            score_threshold=SOFT_NMS_SCORE_THR,
            method="gaussian",
        )

    if len(boxes) == 0:
        return []

    # Step 4: Classification (if classifier is available)
    detections = []

    if classifier.mode != "none":
        # Prepare crops
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        pil_w, pil_h = pil_img.size

        crops = []
        valid_indices = []

        for i, (box, score) in enumerate(zip(boxes, scores)):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
                continue

            valid_indices.append(i)

            # Padded crop for better classification context
            pad_x = w * CROP_PAD_RATIO
            pad_y = h * CROP_PAD_RATIO
            crop = pil_img.crop((
                int(max(0, x1 - pad_x)),
                int(max(0, y1 - pad_y)),
                int(min(pil_w, x2 + pad_x)),
                int(min(pil_h, y2 + pad_y)),
            ))
            crops.append(crop)

        # Batch classification via unified ProductClassifier
        all_classifications = classifier.classify(crops, batch_size=CLASSIFIER_BATCH_SIZE)

        # Step 5: Use detection score for ranking, classifier only for category_id
        # CRITICAL: Do NOT multiply det_score * cls_conf — it destroys mAP ranking
        # by pushing correct detections down when classifier is uncertain
        for idx, (cat_id, cls_conf) in zip(valid_indices, all_classifications):
            x1, y1, x2, y2 = boxes[idx]
            det_score = float(scores[idx])

            detections.append({
                "x1": float(x1),
                "y1": float(y1),
                "w": float(x2 - x1),
                "h": float(y2 - y1),
                "category_id": int(cat_id),
                "score": det_score,
            })
    else:
        # Detection-only mode — category_id=0, use raw detection score
        for box, score in zip(boxes, scores):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
                continue

            detections.append({
                "x1": float(x1),
                "y1": float(y1),
                "w": float(w),
                "h": float(h),
                "category_id": 0,
                "score": float(score),
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
    print(f"[INIT] ONNX detector available: {ONNX_AVAILABLE}")
    print(f"[INIT] SAHI available: {SAHI_AVAILABLE}")
    print(f"[INIT] Soft-NMS available: {SOFT_NMS_AVAILABLE}")
    print(f"[INIT] Ensemble module available: {ENSEMBLE_AVAILABLE}")

    # ── Load detection models ──────────────────────────────────────────
    models = []

    # Discover all model files (.pt and .onnx)
    pt_files = sorted(model_dir.glob("*.pt"))
    onnx_files = sorted(model_dir.glob("*.onnx")) if ONNX_AVAILABLE else []

    # Preferred loading order: best.onnx > best.pt, then secondary models
    primary_loaded = False

    # Try ONNX primary first (e.g. YOLO26 exported)
    primary_onnx = model_dir / "best.onnx"
    if primary_onnx.exists() and ONNX_AVAILABLE:
        try:
            models.append(ONNXDetector(str(primary_onnx), conf_threshold=CONF_THRESHOLD))
            print(f"[MODELS] Loaded primary ONNX detector: {primary_onnx.name}")
            primary_loaded = True
        except Exception as e:
            print(f"[MODELS] Failed to load {primary_onnx.name}: {e}")

    # Fall back to .pt primary
    if not primary_loaded:
        primary_pt = model_dir / "best.pt"
        if primary_pt.exists():
            models.append(YOLO(str(primary_pt)))
            print(f"[MODELS] Loaded primary detector: {primary_pt.name}")
            primary_loaded = True

    if not primary_loaded:
        print("[MODELS] ERROR: No primary model found (best.onnx or best.pt)!")
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output_path), "w") as f:
            json.dump([], f)
        return

    # Load secondary models for ensemble (both .pt and .onnx)
    secondary_names_pt = ["rtdetr_best.pt", "yolo26_best.pt", "yolo11_best.pt"]
    secondary_names_onnx = ["rtdetr_best.onnx", "yolo26_best.onnx", "yolo11_best.onnx"]

    for name in secondary_names_onnx:
        path = model_dir / name
        if path.exists() and ONNX_AVAILABLE:
            try:
                models.append(ONNXDetector(str(path), conf_threshold=CONF_THRESHOLD))
                print(f"[MODELS] Loaded secondary ONNX detector: {path.name}")
            except Exception as e:
                print(f"[MODELS] Failed to load {path.name}: {e}")

    for name in secondary_names_pt:
        path = model_dir / name
        if path.exists():
            try:
                models.append(YOLO(str(path)))
                print(f"[MODELS] Loaded secondary detector: {path.name}")
            except Exception as e:
                print(f"[MODELS] Failed to load {path.name}: {e}")

    # Also pick up any other .onnx files not already loaded
    loaded_names = {"best.onnx"} | set(secondary_names_onnx)
    for onnx_path in onnx_files:
        if onnx_path.name not in loaded_names:
            try:
                models.append(ONNXDetector(str(onnx_path), conf_threshold=CONF_THRESHOLD))
                print(f"[MODELS] Loaded extra ONNX detector: {onnx_path.name}")
            except Exception as e:
                print(f"[MODELS] Failed to load {onnx_path.name}: {e}")

    print(f"[MODELS] Total detectors: {len(models)} ({'ensemble' if len(models) > 1 else 'single'})")

    # ── Load classifier ────────────────────────────────────────────────
    classifier = ProductClassifier(model_dir / "models", device)
    print(f"[INIT] Classification: {classifier.mode}")

    # ── Discover images ────────────────────────────────────────────────
    input_dir = Path(args.input)
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    num_images = len(image_files)
    print(f"[INIT] Found {num_images} images")

    if num_images == 0:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output_path), "w") as f:
            json.dump([], f)
        return

    # ── Time budget ────────────────────────────────────────────────────
    init_time = time.perf_counter() - t_start
    budget = TimeBudget(TOTAL_TIMEOUT - init_time, num_images)
    print(f"[INIT] Setup took {init_time:.1f}s, budget: {budget.total_budget:.0f}s for {num_images} images")

    # ── Process images ─────────────────────────────────────────────────
    predictions = []

    for img_idx, img_path in enumerate(image_files):
        img_start = time.perf_counter()
        images_remaining = num_images - img_idx

        image_id = int(img_path.stem.split("_")[-1])

        # Check time budget
        use_sahi = budget.should_use_sahi(images_remaining)
        if use_sahi:
            budget.sahi_used += 1
        else:
            budget.sahi_skipped += 1

        # Check if we're about to run out of time
        if budget.remaining() < 5:
            print(f"[WARN] Only {budget.remaining():.1f}s remaining — stopping early at image {img_idx}/{num_images}")
            break

        # Read and process
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            print(f"  [{img_idx+1}/{num_images}] SKIP (cannot read): {img_path.name}")
            continue

        detections = process_image(
            models, img_bgr, device, use_sahi, classifier,
        )

        for det in detections:
            predictions.append({
                "image_id": image_id,
                "category_id": det["category_id"],
                "bbox": [
                    round(det["x1"], 1),
                    round(det["y1"], 1),
                    round(det["w"], 1),
                    round(det["h"], 1),
                ],
                "score": round(det["score"], 4),
            })

        img_time = time.perf_counter() - img_start
        budget.record_image(img_time)

        mode = "SAHI" if use_sahi else "fast"
        print(
            f"  [{img_idx+1}/{num_images}] {img_path.name}: "
            f"{len(detections)} dets, {img_time:.2f}s ({mode}) | "
            f"remaining: {budget.remaining():.0f}s"
        )

    # ── Write output ───────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(predictions, f)

    total_time = time.perf_counter() - t_start
    print(f"\n[DONE] {len(predictions)} predictions for {num_images} images in {total_time:.1f}s")
    print(f"[DONE] {budget.summary()}")


if __name__ == "__main__":
    main()
