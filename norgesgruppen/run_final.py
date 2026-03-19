"""NorgesGruppen Object Detection — Final Optimized Submission

Best-in-class pipeline:
  1. CLAHE preprocessing (glare removal from glass doors)
  2. Multi-scale detection (640 + 1280) with TTA
  3. SAHI tiled inference for small products (with time budget)
  4. WBF fusion across scales/tiles (ensemble-boxes library)
  5. Soft-NMS for dense shelf-aware suppression
  6. Hybrid classification: YOLO multi-class + EfficientNet embedding refinement
  7. Adaptive time budget (280s total, auto-disables SAHI if slow)

Scoring: 0.7 * detection_mAP@0.5 + 0.3 * classification_mAP@0.5

No `import os` — uses pathlib only. Sandbox-compatible.

Usage (by sandbox):
    python run.py --input /data/images --output /output/predictions.json
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
from ultralytics import YOLO

from src.utils import enhance_retail_image

# Try using pre-installed ensemble_boxes library (faster C implementation)
try:
    from ensemble_boxes import weighted_boxes_fusion as eb_wbf
    USE_EB_WBF = True
except ImportError:
    USE_EB_WBF = False

# Fallback to our pure-numpy WBF
from src.wbf import weighted_boxes_fusion as custom_wbf

# Optional modules
try:
    from src.soft_nms import soft_nms
    SOFT_NMS_OK = True
except ImportError:
    SOFT_NMS_OK = False

# ── Configuration ──────────────────────────────────────────────────────
TOTAL_TIMEOUT = 280          # leave 20s margin from 300s sandbox limit

# Detection thresholds — very low conf to maximize recall (let scorer sort)
CONF_THRESHOLD = 0.001
NMS_IOU = 0.65
SCALES = [640, 1280]         # multi-scale for WBF

# SAHI tiling
SAHI_TILE_SIZE = 640
SAHI_OVERLAP = 0.25

# WBF
WBF_IOU_THR = 0.55
WBF_SKIP_BOX_THR = 0.001

# Soft-NMS
SOFT_NMS_SIGMA = 0.5
SOFT_NMS_SCORE_THR = 0.005

# Classification
CLASSIFIER_BATCH_SIZE = 64
CROP_PAD_RATIO = 0.05
MIN_BOX_SIZE = 4

# Hybrid classification: when to trust EfficientNet over YOLO
EFF_CONFIDENCE_THRESHOLD = 0.4  # EfficientNet must be this confident to override


def wbf(boxes_list, scores_list, labels_list, weights=None, iou_thr=0.55, skip_box_thr=0.0):
    """WBF wrapper — uses ensemble_boxes library if available, else custom."""
    if USE_EB_WBF:
        return eb_wbf(boxes_list, scores_list, labels_list,
                       weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)
    return custom_wbf(boxes_list, scores_list, labels_list,
                      weights=weights, iou_thr=iou_thr, skip_box_thr=skip_box_thr)


# ── SAHI tiling ────────────────────────────────────────────────────────

def _compute_slices(img_size, slice_size, overlap_px):
    step = slice_size - overlap_px
    slices = []
    pos = 0
    while pos < img_size:
        end = min(pos + slice_size, img_size)
        if end - pos < slice_size and pos > 0:
            pos = max(0, end - slice_size)
        slices.append((pos, min(pos + slice_size, img_size)))
        if end >= img_size:
            break
        pos += step
    return slices


def detect_sahi(model, img, device, img_h, img_w):
    """SAHI: slice image into tiles, detect on each, fuse with WBF."""
    overlap_px = int(SAHI_TILE_SIZE * SAHI_OVERLAP)
    all_boxes, all_scores, all_labels = [], [], []

    # Full image pass at largest scale
    results = model(img, device=device, verbose=False,
                    conf=CONF_THRESHOLD, iou=NMS_IOU, imgsz=SCALES[-1], augment=True)
    for r in results:
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            labels = r.boxes.cls.cpu().numpy().astype(int)
            boxes[:, [0, 2]] /= img_w
            boxes[:, [1, 3]] /= img_h
            all_boxes.append(np.clip(boxes, 0, 1))
            all_scores.append(scores)
            all_labels.append(labels)
        else:
            all_boxes.append(np.zeros((0, 4)))
            all_scores.append(np.array([]))
            all_labels.append(np.array([]))

    # Tiled passes
    x_slices = _compute_slices(img_w, SAHI_TILE_SIZE, overlap_px)
    y_slices = _compute_slices(img_h, SAHI_TILE_SIZE, overlap_px)

    for y_start, y_end in y_slices:
        for x_start, x_end in x_slices:
            tile = img[y_start:y_end, x_start:x_end]
            tile_results = model(tile, device=device, verbose=False,
                                 conf=CONF_THRESHOLD, iou=NMS_IOU, imgsz=SAHI_TILE_SIZE)
            for r in tile_results:
                if r.boxes is None or len(r.boxes) == 0:
                    all_boxes.append(np.zeros((0, 4)))
                    all_scores.append(np.array([]))
                    all_labels.append(np.array([]))
                    continue
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                labels = r.boxes.cls.cpu().numpy().astype(int)
                boxes[:, 0] += x_start
                boxes[:, 2] += x_start
                boxes[:, 1] += y_start
                boxes[:, 3] += y_start
                boxes[:, [0, 2]] /= img_w
                boxes[:, [1, 3]] /= img_h
                all_boxes.append(np.clip(boxes, 0, 1))
                all_scores.append(scores)
                all_labels.append(labels)

    if not any(len(b) > 0 for b in all_boxes):
        return np.zeros((0, 4)), np.array([]), np.array([], dtype=int)

    fused_boxes, fused_scores, fused_labels = wbf(
        all_boxes, all_scores, all_labels,
        weights=[1.0] * len(all_boxes), iou_thr=WBF_IOU_THR, skip_box_thr=WBF_SKIP_BOX_THR)

    # ensemble_boxes returns float labels — cast back to int
    fused_labels = np.asarray(fused_labels, dtype=int)
    fused_boxes = np.asarray(fused_boxes)
    fused_scores = np.asarray(fused_scores)

    if len(fused_boxes) > 0:
        fused_boxes[:, [0, 2]] *= img_w
        fused_boxes[:, [1, 3]] *= img_h

    return fused_boxes, fused_scores, fused_labels


def detect_multiscale(model, img, device, img_h, img_w):
    """Multi-scale WBF with TTA — faster fallback when SAHI is too slow."""
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
            boxes[:, [0, 2]] /= img_w
            boxes[:, [1, 3]] /= img_h
            all_boxes.append(np.clip(boxes, 0, 1))
            all_scores.append(scores)
            all_labels.append(labels)

    if not any(len(b) > 0 for b in all_boxes):
        return np.zeros((0, 4)), np.array([]), np.array([], dtype=int)

    fused_boxes, fused_scores, fused_labels = wbf(
        all_boxes, all_scores, all_labels,
        weights=[1.0] * len(SCALES), iou_thr=WBF_IOU_THR, skip_box_thr=WBF_SKIP_BOX_THR)

    fused_labels = np.asarray(fused_labels, dtype=int)
    fused_boxes = np.asarray(fused_boxes)
    fused_scores = np.asarray(fused_scores)

    if len(fused_boxes) > 0:
        fused_boxes[:, [0, 2]] *= img_w
        fused_boxes[:, [1, 3]] *= img_h

    return fused_boxes, fused_scores, fused_labels


# ── EfficientNet Classifier ───────────────────────────────────────────

class EmbeddingClassifier:
    """EfficientNet-B3 embedding-based product classifier."""

    def __init__(self, model_dir, device):
        self.device = device
        self.model = None
        self.ref_embeddings = None
        self.valid_mask = None
        self.transform = None

        config_path = model_dir / "embedding_config.json"
        embeddings_path = model_dir / "product_embeddings.npy"
        weights_path = model_dir / "efficientnet_b3_weights.pt"

        if not config_path.exists() or not embeddings_path.exists():
            print("[CLASSIFIER] No embedding files found — classification from YOLO only")
            return

        try:
            import timm
            from torchvision import transforms

            with open(str(config_path)) as f:
                config = json.load(f)

            model = timm.create_model(config["model_name"], pretrained=False, num_classes=0)
            if weights_path.exists():
                state_dict = torch.load(str(weights_path), map_location=device, weights_only=True)
                model.load_state_dict(state_dict, strict=False)
                print(f"[CLASSIFIER] Loaded EfficientNet-B3 from {weights_path.name}")
            else:
                print("[CLASSIFIER] WARNING: No weights — embedding matching will be poor")

            model = model.to(device).eval()

            embeddings = np.load(str(embeddings_path))
            ref_embeddings = torch.from_numpy(embeddings).to(device).float()
            valid_mask = ref_embeddings.norm(dim=1) > 0.1

            self.model = model
            self.ref_embeddings = F.normalize(ref_embeddings, dim=1)
            self.valid_mask = valid_mask
            self.transform = transforms.Compose([
                transforms.Resize((300, 300)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            n_valid = int(valid_mask.sum().item())
            print(f"[CLASSIFIER] {n_valid}/{len(ref_embeddings)} categories with embeddings")

        except Exception as e:
            print(f"[CLASSIFIER] Failed to load: {e}")

    @property
    def available(self):
        return self.model is not None

    def classify_batch(self, crops):
        """Classify crops, returns list of (category_id, confidence)."""
        if not self.available or not crops:
            return [(0, 0.0)] * len(crops)

        batch = torch.stack([self.transform(c) for c in crops]).to(self.device)

        with torch.no_grad():
            embeddings = self.model(batch)
            embeddings = F.normalize(embeddings, dim=1)

        similarities = embeddings @ self.ref_embeddings.T
        similarities[:, ~self.valid_mask] = float("-inf")

        # Temperature-scaled softmax
        probs = F.softmax(similarities / 0.07, dim=1)

        results = []
        for i in range(len(crops)):
            best_idx = probs[i].argmax().item()
            best_prob = probs[i, best_idx].item()
            results.append((best_idx, best_prob))

        return results


# ── Main Pipeline ─────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    t_start = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = Path(__file__).parent

    print(f"[INIT] Device: {device}")
    print(f"[INIT] WBF backend: {'ensemble_boxes' if USE_EB_WBF else 'custom'}")
    print(f"[INIT] Soft-NMS: {SOFT_NMS_OK}")

    # Load detector
    model_path = model_dir / "best.pt"
    model = YOLO(str(model_path))
    print(f"[INIT] Loaded detector: {model_path.name}")

    # Load classifier (for hybrid classification refinement)
    classifier = EmbeddingClassifier(model_dir / "models", device)

    # Discover images
    input_dir = Path(args.input)
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )
    n_images = len(image_files)
    print(f"[INIT] {n_images} images to process")

    if n_images == 0:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(str(args.output), "w") as f:
            json.dump([], f)
        return

    # Time budget
    init_time = time.perf_counter() - t_start
    remaining_budget = TOTAL_TIMEOUT - init_time
    print(f"[INIT] Setup: {init_time:.1f}s, budget: {remaining_budget:.0f}s")

    predictions = []
    image_times = []
    sahi_count = 0

    for img_idx, img_path in enumerate(image_files):
        img_start = time.perf_counter()
        images_left = n_images - img_idx
        elapsed = time.perf_counter() - t_start

        # Safety: stop if almost out of time
        time_left = TOTAL_TIMEOUT - elapsed
        if time_left < 5:
            print(f"[WARN] {time_left:.1f}s left — stopping at image {img_idx}/{n_images}")
            break

        image_id = int(img_path.stem.split("_")[-1])

        # Read and CLAHE enhance
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is None:
            continue
        img = enhance_retail_image(img_bgr)
        img_h, img_w = img.shape[:2]

        # Decide: SAHI vs multi-scale based on time budget
        use_sahi = False
        if images_left > 0:
            avg_time = np.mean(image_times) if image_times else 3.0
            time_per_image_budget = time_left / images_left
            # Use SAHI if we have >2x the average time remaining per image
            use_sahi = time_per_image_budget > avg_time * 1.5 and time_left > 30

        # Detect
        if use_sahi:
            boxes, scores, labels = detect_sahi(model, img, device, img_h, img_w)
            sahi_count += 1
        else:
            boxes, scores, labels = detect_multiscale(model, img, device, img_h, img_w)

        if len(boxes) == 0:
            img_time = time.perf_counter() - img_start
            image_times.append(img_time)
            continue

        # Soft-NMS
        if SOFT_NMS_OK and len(boxes) > 0:
            boxes, scores, labels = soft_nms(
                boxes, scores, labels,
                sigma=SOFT_NMS_SIGMA,
                score_threshold=SOFT_NMS_SCORE_THR,
                method="gaussian",
            )

        if len(boxes) == 0:
            img_time = time.perf_counter() - img_start
            image_times.append(img_time)
            continue

        # ── Hybrid classification ──────────────────────────────────────
        # YOLO gives category_id from multi-class training.
        # EfficientNet refines uncertain predictions.
        # If YOLO is single-class (all labels=0), EfficientNet provides all classification.
        is_single_class = len(labels) > 0 and np.all(labels == 0)

        if classifier.available:
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(img_rgb)
            pil_w, pil_h = pil_img.size

            # Batch all crops
            crops = []
            valid_indices = []
            for i, (box, score) in enumerate(zip(boxes, scores)):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
                    continue
                valid_indices.append(i)
                pad_x = w * CROP_PAD_RATIO
                pad_y = h * CROP_PAD_RATIO
                crop = pil_img.crop((
                    int(max(0, x1 - pad_x)),
                    int(max(0, y1 - pad_y)),
                    int(min(pil_w, x2 + pad_x)),
                    int(min(pil_h, y2 + pad_y)),
                ))
                crops.append(crop)

            # Batch classify
            all_eff = []
            for batch_start in range(0, len(crops), CLASSIFIER_BATCH_SIZE):
                batch = crops[batch_start:batch_start + CLASSIFIER_BATCH_SIZE]
                all_eff.extend(classifier.classify_batch(batch))

            # Hybrid: blend YOLO + EfficientNet predictions
            for vi, (eff_cat, eff_conf) in zip(valid_indices, all_eff):
                x1, y1, x2, y2 = boxes[vi]
                det_score = float(scores[vi])
                yolo_cat = int(labels[vi])

                if is_single_class:
                    # Single-class detector: EfficientNet provides ALL classification
                    final_cat = eff_cat
                    final_score = det_score * eff_conf
                elif eff_conf >= EFF_CONFIDENCE_THRESHOLD:
                    if eff_cat == yolo_cat:
                        # Agreement: boost confidence
                        final_cat = yolo_cat
                        final_score = det_score * max(eff_conf, 0.9)
                    elif eff_conf > 0.6:
                        # EfficientNet very confident, disagrees → trust EfficientNet
                        final_cat = eff_cat
                        final_score = det_score * eff_conf
                    else:
                        # Both have some confidence — prefer YOLO (trained e2e)
                        final_cat = yolo_cat
                        final_score = det_score * 0.85
                else:
                    # EfficientNet uncertain — keep YOLO classification
                    final_cat = yolo_cat
                    final_score = det_score * 0.8

                w = x2 - x1
                h = y2 - y1
                predictions.append({
                    "image_id": image_id,
                    "category_id": final_cat,
                    "bbox": [round(float(x1), 1), round(float(y1), 1),
                             round(float(w), 1), round(float(h), 1)],
                    "score": round(final_score, 4),
                })
        else:
            # No classifier — use YOLO multi-class predictions directly
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
                    continue
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(label),
                    "bbox": [round(float(x1), 1), round(float(y1), 1),
                             round(float(w), 1), round(float(h), 1)],
                    "score": round(float(score), 4),
                })

        img_time = time.perf_counter() - img_start
        image_times.append(img_time)

        mode = "SAHI" if use_sahi else "MS"
        det_count = sum(1 for p in predictions if p["image_id"] == image_id)
        if (img_idx + 1) % 10 == 0 or img_idx < 3:
            print(f"  [{img_idx+1}/{n_images}] {img_path.name}: {det_count} dets, "
                  f"{img_time:.2f}s ({mode}) | {TOTAL_TIMEOUT - (time.perf_counter() - t_start):.0f}s left")

    # Write output
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(predictions, f)

    total = time.perf_counter() - t_start
    avg_t = np.mean(image_times) if image_times else 0
    print(f"\n[DONE] {len(predictions)} predictions for {len(image_times)}/{n_images} images")
    print(f"[DONE] {total:.1f}s total, {avg_t:.2f}s/img avg, SAHI: {sahi_count} images")


if __name__ == "__main__":
    main()
