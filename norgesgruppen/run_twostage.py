"""NorgesGruppen Object Detection — Two-Stage: Detect + Classify

Stage 1: Single-class YOLOv8x detector with multi-scale WBF ensemble
Stage 2: EfficientNet-B3 embedding matching (identifies which product)

Scoring: 70% detection mAP + 30% classification mAP
Detection is weighted higher, so a good detector is critical.

No `import os` — uses pathlib only. Sandbox-compatible.

Usage (by sandbox):
    python run.py --input /data/images --output /output/predictions.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from ultralytics import YOLO

from src.utils import enhance_retail_image

# Detection config — single scale to stay within 300s timeout
# Multi-scale WBF is ~2x slower and risks timeout with 250+ images
IMGSZ = 1280
CONF_THRESHOLD = 0.15
NMS_IOU = 0.45


def load_classifier(model_dir: Path, device: str):
    """Load EfficientNet-B3 and reference embeddings."""
    import timm

    config_path = model_dir / "embedding_config.json"
    with open(str(config_path)) as f:
        config = json.load(f)

    # Load fine-tuned weights (required — sandbox has no network for pretrained download)
    weights_path = model_dir / "efficientnet_b3_weights.pt"
    model = timm.create_model(config["model_name"], pretrained=False, num_classes=0)
    if weights_path.exists():
        state_dict = torch.load(str(weights_path), map_location=device)
        model.load_state_dict(state_dict, strict=False)
    else:
        print("WARNING: efficientnet_b3_weights.pt not found! Classification will be random.")

    model = model.to(device).eval()

    embeddings = np.load(str(model_dir / "product_embeddings.npy"))
    ref_embeddings = torch.from_numpy(embeddings).to(device)
    valid_mask = ref_embeddings.norm(dim=1) > 0.1

    transform = transforms.Compose([
        transforms.Resize((300, 300)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    return model, ref_embeddings, valid_mask, transform, config


def classify_crops(
    model, ref_embeddings, valid_mask, transform, crops: list, device: str,
    temperature: float = 0.07,
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

    probs = F.softmax(similarities / temperature, dim=1)

    results = []
    for i in range(len(crops)):
        best_idx = probs[i].argmax().item()
        best_prob = probs[i, best_idx].item()
        results.append((best_idx, best_prob))

    return results


def run_detection(detector, img, device: str):
    """Run single-class detection. Returns pixel-coord boxes and scores."""
    results = detector(
        img,
        device=device,
        verbose=False,
        conf=CONF_THRESHOLD,
        iou=NMS_IOU,
        imgsz=IMGSZ,
        augment=True,
    )

    boxes_list = []
    scores_list = []

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue
        boxes_list.append(r.boxes.xyxy.cpu().numpy())
        scores_list.append(r.boxes.conf.cpu().numpy())

    if not boxes_list:
        return np.zeros((0, 4)), np.array([])

    return np.concatenate(boxes_list), np.concatenate(scores_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = Path(__file__).parent

    # Stage 1: Load single-class detector
    detector_path = model_dir / "best.pt"
    detector = YOLO(str(detector_path))

    # Stage 2: Load classifier
    classifier, ref_embeddings, valid_mask, transform, config = load_classifier(
        model_dir, device
    )

    predictions = []
    input_dir = Path(args.input)
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for img_path in image_files:
        image_id = int(img_path.stem.split("_")[-1])

        # CLAHE preprocessing
        img_bgr = cv2.imread(str(img_path))
        img_enhanced = enhance_retail_image(img_bgr)

        # Stage 1: Single-class detection
        det_boxes, det_scores = run_detection(detector, img_enhanced, device)

        if len(det_boxes) == 0:
            continue

        # Prepare crops for classification
        img_rgb = cv2.cvtColor(img_enhanced, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        img_w, img_h = pil_img.size

        crops = []
        valid_detections = []

        for box, det_score in zip(det_boxes, det_scores):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            if w < 5 or h < 5:
                continue

            valid_detections.append({
                "x1": float(x1), "y1": float(y1),
                "w": float(w), "h": float(h),
                "det_conf": float(det_score),
            })

            # Padded crop for better classification
            pad_x = w * 0.05
            pad_y = h * 0.05
            crop = pil_img.crop((
                int(max(0, x1 - pad_x)),
                int(max(0, y1 - pad_y)),
                int(min(img_w, x2 + pad_x)),
                int(min(img_h, y2 + pad_y)),
            ))
            crops.append(crop)

        # Stage 2: Classify crops in batches
        batch_size = 64
        all_classifications = []
        for batch_start in range(0, len(crops), batch_size):
            batch_crops = crops[batch_start:batch_start + batch_size]
            classifications = classify_crops(
                classifier, ref_embeddings, valid_mask, transform,
                batch_crops, device,
            )
            all_classifications.extend(classifications)

        # Combine detection + classification
        for det, (cat_id, cls_conf) in zip(valid_detections, all_classifications):
            combined_score = det["det_conf"] * cls_conf

            predictions.append({
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": [
                    round(det["x1"], 1),
                    round(det["y1"], 1),
                    round(det["w"], 1),
                    round(det["h"], 1),
                ],
                "score": round(combined_score, 4),
            })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions for {len(image_files)} images")


if __name__ == "__main__":
    main()
