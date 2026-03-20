"""NorgesGruppen Object Detection — ONNX-based Two-Stage Pipeline

Stage 1: YOLO26-x single-class detector (ONNX, FP16)
Stage 2: EfficientNet-B3 finetuned classifier (embedding matching)

Uses onnxruntime-gpu for detection (no ultralytics dependency for YOLO26).
Sandbox-compatible: no `import os`, uses pathlib only.

Usage (by sandbox):
    python run.py --input /data/images --output /output/predictions.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

# Detection config
IMGSZ = 1280
CONF_THRESHOLD = 0.15
NMS_IOU = 0.45


def letterbox(img, new_shape=(1280, 1280), color=(114, 114, 114)):
    """Resize image with letterbox padding (same as YOLO preprocessing)."""
    shape = img.shape[:2]  # h, w
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))
    dw = (new_shape[1] - new_unpad[0]) / 2
    dh = (new_shape[0] - new_unpad[1]) / 2

    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

    return img, r, (dw, dh)


def nms(boxes, scores, iou_threshold=0.45):
    """Non-maximum suppression."""
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)

    order = scores.argsort()[::-1]
    keep = []

    while len(order) > 0:
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break

        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0, xx2 - xx1)
        h = np.maximum(0, yy2 - yy1)
        inter = w * h
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)

        mask = iou <= iou_threshold
        order = order[1:][mask]

    return keep


def preprocess_image(img_bgr, imgsz=1280):
    """Preprocess image for YOLO ONNX inference."""
    img_lb, ratio, (dw, dh) = letterbox(img_bgr, (imgsz, imgsz))
    img_rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = np.transpose(img_norm, (2, 0, 1))
    img_batch = np.expand_dims(img_chw, axis=0)
    return img_batch, ratio, (dw, dh)


def postprocess_detections(output, ratio, dw, dh, conf_threshold=0.15, iou_threshold=0.45):
    """Post-process YOLO26 ONNX output to get boxes in original image coords.

    YOLO26 output format: [1, 300, 6] where last dim is [x1, y1, x2, y2, confidence, class_id]
    Already in xyxy corner format, NMS-free (built into model).
    """
    preds = output[0]  # First output tensor

    if len(preds.shape) == 3:
        preds = preds[0]  # Remove batch dim: [300, 6]

    # Filter by confidence (col 4)
    scores = preds[:, 4]
    mask = scores > conf_threshold
    preds = preds[mask]
    scores = scores[mask]

    if len(preds) == 0:
        return np.zeros((0, 4)), np.array([])

    # Already in xyxy format
    boxes = preds[:, :4].copy()

    # Undo letterbox: remove padding and scale
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - dw) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - dh) / ratio

    # Clip to image bounds
    boxes = np.clip(boxes, 0, None)

    return boxes, scores


def enhance_retail_image(img_bgr):
    """CLAHE enhancement for retail shelf images."""
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)


def load_classifier(model_dir, device):
    """Load EfficientNet-B3 and reference embeddings."""
    import timm

    config_path = model_dir / "embedding_config.json"
    with open(str(config_path)) as f:
        config = json.load(f)

    weights_path = model_dir / "efficientnet_b3_weights.pt"
    model = timm.create_model(config["model_name"], pretrained=False, num_classes=0)
    if weights_path.exists():
        state_dict = torch.load(str(weights_path), map_location=device)
        model.load_state_dict(state_dict, strict=False)

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


def classify_crops(model, ref_embeddings, valid_mask, transform, crops, device, temperature=0.07):
    """Classify cropped product images via embedding similarity."""
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model_dir = Path(__file__).parent

    # Stage 1: Load ONNX detector
    onnx_path = model_dir / "yolo26x_single_best.onnx"
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"] if device == "cuda" else ["CPUExecutionProvider"]
    detector = ort.InferenceSession(str(onnx_path), providers=providers)
    input_name = detector.get_inputs()[0].name

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

        # Stage 1: ONNX detection
        img_batch, ratio, (dw, dh) = preprocess_image(img_enhanced, IMGSZ)
        output = detector.run(None, {input_name: img_batch})
        det_boxes, det_scores = postprocess_detections(
            output, ratio, dw, dh, CONF_THRESHOLD, NMS_IOU
        )

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

    print(f"Wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
