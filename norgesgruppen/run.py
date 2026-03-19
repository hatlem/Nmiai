"""NorgesGruppen Object Detection — Submission run.py

Runs YOLOv8x with CLAHE preprocessing + multi-scale WBF ensemble.
No `import os` — uses pathlib only.

Usage (by sandbox):
    python run.py --input /data/images --output /output/predictions.json
"""

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.utils import enhance_retail_image
from src.wbf import weighted_boxes_fusion

# Multi-scale WBF config (tuned for mAP@0.5 — let scorer sort precision/recall)
SCALES = [640, 1280]
WBF_IOU_THR = 0.6
WBF_SKIP_BOX_THR = 0.001
CONF_THRESHOLD = 0.001
NMS_IOU = 0.65



def run_multiscale_wbf(model, img, device: str):
    """Run inference at multiple scales and fuse with WBF.

    Args:
        model: YOLO model
        img: BGR numpy array (already CLAHE-enhanced)
        device: cuda or cpu

    Returns:
        fused_boxes: (N, 4) in pixel coords [x1, y1, x2, y2]
        fused_scores: (N,)
        fused_labels: (N,) int category ids
    """
    img_h, img_w = img.shape[:2]

    all_boxes = []
    all_scores = []
    all_labels = []

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

            # Normalize to 0-1 (required by WBF)
            boxes[:, 0] /= img_w
            boxes[:, 1] /= img_h
            boxes[:, 2] /= img_w
            boxes[:, 3] /= img_h
            boxes = np.clip(boxes, 0, 1)

            all_boxes.append(boxes)
            all_scores.append(scores)
            all_labels.append(labels)

    # WBF fusion
    if not any(len(b) > 0 for b in all_boxes):
        return np.zeros((0, 4)), np.array([]), np.array([])

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        all_boxes,
        all_scores,
        all_labels,
        weights=[1.0] * len(SCALES),
        iou_thr=WBF_IOU_THR,
        skip_box_thr=WBF_SKIP_BOX_THR,
    )

    # De-normalize back to pixel coords
    fused_boxes[:, 0] *= img_w
    fused_boxes[:, 1] *= img_h
    fused_boxes[:, 2] *= img_w
    fused_boxes[:, 3] *= img_h

    return fused_boxes, fused_scores, fused_labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = Path(__file__).parent / "best.pt"
    model = YOLO(str(model_path))

    predictions = []

    input_dir = Path(args.input)
    image_files = sorted(
        p for p in input_dir.iterdir()
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for img_path in image_files:
        image_id = int(img_path.stem.split("_")[-1])

        # CLAHE preprocessing
        img = cv2.imread(str(img_path))
        img = enhance_retail_image(img)

        # Multi-scale WBF ensemble
        fused_boxes, fused_scores, fused_labels = run_multiscale_wbf(
            model, img, device
        )

        for box, score, label in zip(fused_boxes, fused_scores, fused_labels):
            x1, y1, x2, y2 = box
            w = x2 - x1
            h = y2 - y1

            if w < 3 or h < 3:
                continue

            predictions.append({
                "image_id": image_id,
                "category_id": int(label),
                "bbox": [round(x1, 1), round(y1, 1), round(w, 1), round(h, 1)],
                "score": round(float(score), 4),
            })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(str(output_path), "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions for {len(image_files)} images")


if __name__ == "__main__":
    main()
