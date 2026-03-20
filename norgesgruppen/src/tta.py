"""Test-Time Augmentation (TTA) with Weighted Boxes Fusion.

Multi-scale + horizontal flip inference fused via WBF.
No `import os` — uses pathlib only. Sandbox-safe.
"""

import numpy as np
import cv2

from src.wbf import weighted_boxes_fusion


# Scales to run inference at (width=height square input).
TTA_SCALES = [640, 960, 1280]

# WBF parameters
WBF_IOU_THR = 0.55
WBF_SKIP_BOX_THR = 0.001


def _run_detector(detector, img_bgr: np.ndarray, conf: float, iou: float, imgsz: int):
    """Run detector and return (boxes_xyxy, scores, labels) in pixel coords.

    Works with both ONNXDetector (has _infer) and ultralytics YOLO models.
    """
    # Fast path: ONNXDetector exposes _infer directly
    if hasattr(detector, "_infer"):
        boxes, scores, labels = detector._infer(img_bgr, imgsz, conf, iou)
        return boxes, scores, labels

    # Fallback: ultralytics YOLO interface
    results = detector(img_bgr, conf=conf, iou=iou, imgsz=imgsz)
    boxes_obj = results[0].boxes
    if boxes_obj is None or len(boxes_obj) == 0:
        return np.zeros((0, 4)), np.array([]), np.array([], dtype=int)

    boxes = boxes_obj.xyxy.cpu().numpy() if hasattr(boxes_obj.xyxy, "cpu") else boxes_obj.xyxy.numpy()
    scores = boxes_obj.conf.cpu().numpy() if hasattr(boxes_obj.conf, "cpu") else boxes_obj.conf.numpy()
    # YOLO .pt models have .cls for class IDs
    if hasattr(boxes_obj, "cls") and boxes_obj.cls is not None:
        labels = boxes_obj.cls.cpu().numpy().astype(int) if hasattr(boxes_obj.cls, "cpu") else boxes_obj.cls.numpy().astype(int)
    else:
        labels = np.zeros(len(scores), dtype=int)

    return boxes, scores, labels


def _flip_boxes_horizontal(boxes: np.ndarray, img_w: int) -> np.ndarray:
    """Mirror xyxy boxes horizontally back to original orientation."""
    if len(boxes) == 0:
        return boxes
    flipped = boxes.copy()
    flipped[:, 0] = img_w - boxes[:, 2]
    flipped[:, 2] = img_w - boxes[:, 0]
    return flipped


def tta_detect(
    detector,
    img_bgr: np.ndarray,
    conf: float = 0.01,
    iou: float = 0.65,
    scales: list[int] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Multi-scale TTA with WBF fusion.

    Runs detector at multiple scales and with horizontal flip,
    then fuses all predictions using Weighted Boxes Fusion.

    Args:
        detector: ONNXDetector or YOLO model (callable with ultralytics interface)
        img_bgr: BGR image (H, W, 3)
        conf: confidence threshold
        iou: NMS IoU threshold
        scales: list of input sizes (default: [640, 960, 1280])

    Returns:
        boxes (N, 4) xyxy in pixel coords, scores (N,), labels (N,)
    """
    if scales is None:
        scales = TTA_SCALES

    h, w = img_bgr.shape[:2]

    # Collect predictions from all augmentations
    all_boxes = []
    all_scores = []
    all_labels = []
    # Weights per augmentation: larger scales get slightly more weight
    all_weights = []

    # Pre-flip the image once
    flipped_img = cv2.flip(img_bgr, 1)

    for scale in scales:
        # Weight: proportional to scale (1280 gets more weight than 640)
        weight = scale / 1280.0

        # --- Original orientation ---
        boxes, scores, labels = _run_detector(detector, img_bgr, conf, iou, scale)
        if len(boxes) > 0:
            # Normalize to [0, 1]
            norm_boxes = boxes.copy().astype(np.float64)
            norm_boxes[:, [0, 2]] /= w
            norm_boxes[:, [1, 3]] /= h
            norm_boxes = np.clip(norm_boxes, 0.0, 1.0)
            all_boxes.append(norm_boxes.astype(np.float32))
            all_scores.append(scores.astype(np.float32))
            all_labels.append(labels.astype(int))
        else:
            all_boxes.append(np.zeros((0, 4), dtype=np.float32))
            all_scores.append(np.array([], dtype=np.float32))
            all_labels.append(np.array([], dtype=int))
        all_weights.append(weight)

        # --- Horizontal flip ---
        f_boxes, f_scores, f_labels = _run_detector(detector, flipped_img, conf, iou, scale)
        if len(f_boxes) > 0:
            # Mirror boxes back to original coordinate space
            f_boxes = _flip_boxes_horizontal(f_boxes, w)
            # Normalize to [0, 1]
            norm_boxes = f_boxes.copy().astype(np.float64)
            norm_boxes[:, [0, 2]] /= w
            norm_boxes[:, [1, 3]] /= h
            norm_boxes = np.clip(norm_boxes, 0.0, 1.0)
            all_boxes.append(norm_boxes.astype(np.float32))
            all_scores.append(f_scores.astype(np.float32))
            all_labels.append(f_labels.astype(int))
        else:
            all_boxes.append(np.zeros((0, 4), dtype=np.float32))
            all_scores.append(np.array([], dtype=np.float32))
            all_labels.append(np.array([], dtype=int))
        all_weights.append(weight)

    # Check if we have any predictions at all
    total_preds = sum(len(b) for b in all_boxes)
    if total_preds == 0:
        return np.zeros((0, 4)), np.array([]), np.array([], dtype=int)

    # Fuse with WBF
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        all_boxes,
        all_scores,
        all_labels,
        weights=all_weights,
        iou_thr=WBF_IOU_THR,
        skip_box_thr=WBF_SKIP_BOX_THR,
    )

    if len(fused_boxes) == 0:
        return np.zeros((0, 4)), np.array([]), np.array([], dtype=int)

    # Convert back to pixel coordinates
    pixel_boxes = fused_boxes.copy()
    pixel_boxes[:, [0, 2]] *= w
    pixel_boxes[:, [1, 3]] *= h

    return pixel_boxes, fused_scores, fused_labels
