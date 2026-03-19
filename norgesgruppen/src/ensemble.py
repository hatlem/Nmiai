"""Multi-model ensemble inference with Weighted Boxes Fusion.

Loads multiple ultralytics models (YOLOv8x + RT-DETR) and fuses their
predictions using WBF for improved detection performance.

No `import os` — uses pathlib only. Sandbox-safe.
"""

import numpy as np
from ultralytics import YOLO

from src.wbf import weighted_boxes_fusion


def _run_single_model(
    model,
    img: np.ndarray,
    device: str,
    imgsz: int,
    conf_threshold: float,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run inference with a single model and return normalized boxes.

    Returns:
        boxes: (N, 4) normalized to [0, 1] as [x1, y1, x2, y2]
        scores: (N,) confidence scores
        labels: (N,) integer class ids
    """
    img_h, img_w = img.shape[:2]

    results = model(
        img,
        device=device,
        verbose=False,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=imgsz,
        augment=True,
    )

    all_boxes = []
    all_scores = []
    all_labels = []

    for r in results:
        if r.boxes is None or len(r.boxes) == 0:
            continue

        boxes = r.boxes.xyxy.cpu().numpy()
        scores = r.boxes.conf.cpu().numpy()
        labels = r.boxes.cls.cpu().numpy().astype(int)

        # Normalize to [0, 1] for WBF
        boxes[:, 0] /= img_w
        boxes[:, 1] /= img_h
        boxes[:, 2] /= img_w
        boxes[:, 3] /= img_h
        boxes = np.clip(boxes, 0, 1)

        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    if not all_boxes:
        return np.zeros((0, 4)), np.array([]), np.array([], dtype=int)

    return (
        np.concatenate(all_boxes),
        np.concatenate(all_scores),
        np.concatenate(all_labels),
    )


def ensemble_inference(
    models: list,
    img: np.ndarray,
    device: str,
    scales: list[int] | None = None,
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.45,
    wbf_iou_thr: float = 0.55,
    wbf_skip_box_thr: float = 0.001,
    weights: list[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run ensemble inference across multiple models and fuse with WBF.

    Args:
        models: List of loaded ultralytics model objects (YOLO or RTDETR).
        img: BGR numpy array (already preprocessed, e.g. CLAHE-enhanced).
        device: "cuda" or "cpu".
        scales: Per-model inference resolution. Default [1280] for each model.
        conf_threshold: Minimum confidence for detections.
        iou_threshold: NMS IoU threshold per model.
        wbf_iou_thr: WBF IoU threshold for merging boxes across models.
        wbf_skip_box_thr: WBF minimum score to keep a box.
        weights: Per-model weights for WBF fusion. Default equal weights.

    Returns:
        boxes_xyxy: (N, 4) fused boxes in pixel coordinates [x1, y1, x2, y2].
        scores: (N,) fused confidence scores.
        class_ids: (N,) integer class labels.
    """
    n_models = len(models)
    if scales is None:
        scales = [1280] * n_models
    if weights is None:
        weights = [1.0] * n_models

    assert len(scales) == n_models, "scales must match number of models"
    assert len(weights) == n_models, "weights must match number of models"

    img_h, img_w = img.shape[:2]

    boxes_list = []
    scores_list = []
    labels_list = []

    for model, scale in zip(models, scales):
        boxes, scores, labels = _run_single_model(
            model, img, device, scale, conf_threshold, iou_threshold,
        )
        boxes_list.append(boxes)
        scores_list.append(scores)
        labels_list.append(labels)

    # Check if any model produced detections
    if not any(len(b) > 0 for b in boxes_list):
        return np.zeros((0, 4)), np.array([]), np.array([], dtype=int)

    # WBF fusion (expects normalized [0, 1] boxes)
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=weights,
        iou_thr=wbf_iou_thr,
        skip_box_thr=wbf_skip_box_thr,
    )

    # De-normalize back to pixel coordinates
    if len(fused_boxes) > 0:
        fused_boxes[:, 0] *= img_w
        fused_boxes[:, 1] *= img_h
        fused_boxes[:, 2] *= img_w
        fused_boxes[:, 3] *= img_h

    return fused_boxes, fused_scores, fused_labels


def load_ensemble(
    model_paths: list[str],
) -> list:
    """Load multiple ultralytics models for ensemble inference.

    Args:
        model_paths: List of paths to .pt weight files.
            Supports YOLOv8 and RT-DETR models from ultralytics==8.1.0.

    Returns:
        List of loaded model objects.
    """
    models = []
    for path in model_paths:
        model = YOLO(path)  # YOLO() loads both YOLOv8 and RT-DETR .pt files
        models.append(model)
        print(f"Loaded: {path}")
    return models
