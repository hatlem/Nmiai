"""NorgesGruppen Object Detection — Best Combined Submission

Integrates all improvements:
  1. CLAHE preprocessing (glare removal)
  2. SAHI tiled inference (small object detection)
  3. Multi-model ensemble (if multiple models exist)
  4. Soft-NMS (dense shelf-aware suppression)
  5. Two-stage classification (EfficientNet-B3 embedding matching)
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
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from src.utils import enhance_retail_image
from src.wbf import weighted_boxes_fusion

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
CONF_THRESHOLD = 0.10
NMS_IOU = 0.45
IMGSZ_FULL = 1280
IMGSZ_SAHI_TILE = 640
SAHI_OVERLAP = 0.2

# Multi-scale WBF (fallback when SAHI unavailable)
SCALES = [640, 1280]
WBF_IOU_THR = 0.55
WBF_SKIP_BOX_THR = 0.001

# Soft-NMS
SOFT_NMS_SIGMA = 0.5
SOFT_NMS_SCORE_THR = 0.01

# Classification
CLASSIFIER_BATCH_SIZE = 64
TEMPERATURE = 0.07
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


# ── Classifier Loading ────────────────────────────────────────────────

def load_classifier(model_dir: Path, device: str):
    """Load EfficientNet-B3 classifier and reference embeddings.

    Returns None tuple if classifier files are missing.
    """
    config_path = model_dir / "embedding_config.json"
    embeddings_path = model_dir / "product_embeddings.npy"

    if not config_path.exists() or not embeddings_path.exists():
        print("[CLASSIFIER] Config or embeddings not found — detection-only mode")
        return None, None, None, None, None

    try:
        import timm
    except ImportError:
        print("[CLASSIFIER] timm not available — detection-only mode")
        return None, None, None, None, None

    with open(str(config_path)) as f:
        config = json.load(f)

    weights_path = model_dir / "efficientnet_b3_weights.pt"
    model = timm.create_model(config["model_name"], pretrained=False, num_classes=0)

    if weights_path.exists():
        state_dict = torch.load(str(weights_path), map_location=device)
        model.load_state_dict(state_dict, strict=False)
        print(f"[CLASSIFIER] Loaded fine-tuned weights from {weights_path.name}")
    else:
        print("[CLASSIFIER] WARNING: No fine-tuned weights — classification will be poor")

    model = model.to(device).eval()

    embeddings = np.load(str(embeddings_path))
    ref_embeddings = torch.from_numpy(embeddings).to(device)
    valid_mask = ref_embeddings.norm(dim=1) > 0.1

    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    print(f"[CLASSIFIER] Loaded {ref_embeddings.shape[0]} reference embeddings")
    return model, ref_embeddings, valid_mask, transform, config


def classify_crops(
    model, ref_embeddings, valid_mask, transform, crops: list, device: str,
) -> list[tuple[int, float]]:
    """Classify cropped product images via temperature-scaled embedding similarity."""
    if not crops:
        return []

    batch = torch.stack([transform(crop) for crop in crops]).to(device)

    with torch.no_grad():
        embeddings = model(batch)
        embeddings = F.normalize(embeddings, dim=1)

    similarities = embeddings @ ref_embeddings.T
    similarities[:, ~valid_mask] = float("-inf")

    probs = F.softmax(similarities / TEMPERATURE, dim=1)

    results = []
    for i in range(len(crops)):
        best_idx = probs[i].argmax().item()
        best_prob = probs[i, best_idx].item()
        results.append((best_idx, best_prob))

    return results


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
        augment=True,
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
            augment=True,
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
    classifier_components: tuple,
) -> list[dict]:
    """Full pipeline for a single image.

    1. CLAHE enhance
    2. Detect (SAHI/ensemble/multi-scale/simple — best available)
    3. Soft-NMS on merged detections
    4. Classify crops with EfficientNet-B3
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
    elif SAHI_AVAILABLE:
        # Even without ensemble, SAHI alone is better than multi-scale
        boxes, scores, labels = detect_sahi(primary_model, img, device)
    else:
        # Fallback: multi-scale WBF
        boxes, scores, labels = detect_multiscale_wbf(primary_model, img, device)

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
    cls_model, ref_embeddings, valid_mask, cls_transform, cls_config = classifier_components

    detections = []

    if cls_model is not None:
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

        # Batch classification
        all_classifications = []
        for batch_start in range(0, len(crops), CLASSIFIER_BATCH_SIZE):
            batch_crops = crops[batch_start:batch_start + CLASSIFIER_BATCH_SIZE]
            classifications = classify_crops(
                cls_model, ref_embeddings, valid_mask, cls_transform,
                batch_crops, device,
            )
            all_classifications.extend(classifications)

        # Step 5: Combine detection score x classification confidence
        for idx, (cat_id, cls_conf) in zip(valid_indices, all_classifications):
            x1, y1, x2, y2 = boxes[idx]
            det_score = float(scores[idx])
            combined_score = det_score * cls_conf

            detections.append({
                "x1": float(x1),
                "y1": float(y1),
                "w": float(x2 - x1),
                "h": float(y2 - y1),
                "category_id": int(cat_id),
                "score": combined_score,
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
    print(f"[INIT] SAHI available: {SAHI_AVAILABLE}")
    print(f"[INIT] Soft-NMS available: {SOFT_NMS_AVAILABLE}")
    print(f"[INIT] Ensemble module available: {ENSEMBLE_AVAILABLE}")

    # ── Load detection models ──────────────────────────────────────────
    models = []

    primary_path = model_dir / "best.pt"
    if primary_path.exists():
        models.append(YOLO(str(primary_path)))
        print(f"[MODELS] Loaded primary detector: {primary_path.name}")
    else:
        print("[MODELS] ERROR: best.pt not found!")
        # Write empty predictions
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(str(output_path), "w") as f:
            json.dump([], f)
        return

    # Try loading secondary model for ensemble
    secondary_path = model_dir / "rtdetr_best.pt"
    if secondary_path.exists():
        try:
            models.append(YOLO(str(secondary_path)))
            print(f"[MODELS] Loaded secondary detector: {secondary_path.name}")
        except Exception as e:
            print(f"[MODELS] Failed to load {secondary_path.name}: {e}")

    print(f"[MODELS] Total detectors: {len(models)} ({'ensemble' if len(models) > 1 else 'single'})")

    # ── Load classifier ────────────────────────────────────────────────
    classifier_components = load_classifier(model_dir, device)
    has_classifier = classifier_components[0] is not None
    print(f"[INIT] Classification: {'enabled' if has_classifier else 'detection-only (category_id=0)'}")

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
            models, img_bgr, device, use_sahi, classifier_components,
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
