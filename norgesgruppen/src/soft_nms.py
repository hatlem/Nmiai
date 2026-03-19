"""Soft-NMS implementation for dense grocery shelf object detection.

Soft-NMS decays detection scores based on IoU overlap instead of hard
elimination, preserving more overlapping detections — critical for densely
packed products on shelves.

Reference: Bodla et al., "Soft-NMS — Improving Object Detection With One
Line of Code", ICCV 2017.

Pure numpy, no `import os` (sandbox restriction).
"""

import numpy as np


def _iou_single(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Compute IoU between one box and an array of boxes.

    Args:
        box: (4,) [x1, y1, x2, y2]
        boxes: (M, 4) [x1, y1, x2, y2]

    Returns:
        (M,) IoU values
    """
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    intersection = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)

    area_box = (box[2] - box[0]) * (box[3] - box[1])
    area_boxes = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    union = area_box + area_boxes - intersection
    return np.where(union > 0, intersection / union, 0.0)


def _soft_nms_single_class(
    boxes: np.ndarray,
    scores: np.ndarray,
    sigma: float = 0.5,
    score_threshold: float = 0.01,
    iou_threshold: float = 0.3,
    method: str = "gaussian",
) -> tuple[np.ndarray, np.ndarray]:
    """Run Soft-NMS on a single class.

    Args:
        boxes: (N, 4) [x1, y1, x2, y2]
        scores: (N,)
        sigma: Gaussian decay parameter (lower = more aggressive suppression)
        score_threshold: Remove boxes with score below this after decay
        iou_threshold: IoU threshold for linear method hard cutoff
        method: "gaussian" or "linear"

    Returns:
        (keep_indices, decayed_scores) — indices into the original arrays
        and their updated scores.
    """
    n = len(boxes)
    if n == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.float64)

    # Work on copies
    boxes = boxes.copy()
    scores = scores.copy().astype(np.float64)
    indices = np.arange(n)

    # Result lists
    keep_indices = []
    keep_scores = []

    for _ in range(n):
        # Find the detection with max score among remaining
        max_pos = np.argmax(scores)
        max_score = scores[max_pos]

        if max_score < score_threshold:
            break

        # Record this detection
        keep_indices.append(indices[max_pos])
        keep_scores.append(max_score)

        # Swap max to front then remove it from consideration
        current_box = boxes[max_pos].copy()

        # Remove picked box by swapping with last and shrinking
        last = len(scores) - 1
        boxes[max_pos] = boxes[last]
        scores[max_pos] = scores[last]
        indices[max_pos] = indices[last]
        boxes = boxes[:last]
        scores = scores[:last]
        indices = indices[:last]

        if len(scores) == 0:
            break

        # Compute IoU of remaining boxes with the picked box
        ious = _iou_single(current_box, boxes)

        # Decay scores
        if method == "gaussian":
            scores *= np.exp(-(ious ** 2) / sigma)
        elif method == "linear":
            decay = np.where(ious > iou_threshold, 1.0 - ious, 1.0)
            scores *= decay
        else:
            raise ValueError(f"Unknown method: {method!r}. Use 'gaussian' or 'linear'.")

        # Prune boxes that fell below threshold
        mask = scores >= score_threshold
        boxes = boxes[mask]
        scores = scores[mask]
        indices = indices[mask]

    if len(keep_indices) == 0:
        return np.array([], dtype=np.intp), np.array([], dtype=np.float64)

    return np.array(keep_indices, dtype=np.intp), np.array(keep_scores, dtype=np.float64)


def soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    sigma: float = 0.5,
    score_threshold: float = 0.01,
    iou_threshold: float = 0.3,
    method: str = "gaussian",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Soft-NMS with per-class suppression.

    For each unique class label, Soft-NMS is applied independently.
    This prevents cross-class suppression (e.g., a milk carton shouldn't
    suppress an overlapping juice box).

    Args:
        boxes: (N, 4) in [x1, y1, x2, y2] format (pixel coordinates).
        scores: (N,) confidence scores.
        labels: (N,) integer class labels.
        sigma: Gaussian decay parameter. Lower values = more aggressive
            suppression. 0.5 is a good default for dense shelves.
        score_threshold: Discard detections with decayed score below this.
        iou_threshold: IoU threshold used only by the "linear" method.
        method: "gaussian" (recommended for dense scenes) or "linear".

    Returns:
        Tuple of (filtered_boxes, filtered_scores, filtered_labels), each
        as numpy arrays with consistent length.
    """
    if len(boxes) == 0:
        return (
            np.zeros((0, 4), dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
        )

    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)

    unique_labels = np.unique(labels)

    all_boxes = []
    all_scores = []
    all_labels = []

    for cls in unique_labels:
        cls_mask = labels == cls
        cls_boxes = boxes[cls_mask]
        cls_scores = scores[cls_mask]

        keep_idx, keep_scores = _soft_nms_single_class(
            cls_boxes,
            cls_scores,
            sigma=sigma,
            score_threshold=score_threshold,
            iou_threshold=iou_threshold,
            method=method,
        )

        if len(keep_idx) > 0:
            all_boxes.append(cls_boxes[keep_idx])
            all_scores.append(keep_scores)
            all_labels.append(np.full(len(keep_idx), cls, dtype=np.int64))

    if not all_boxes:
        return (
            np.zeros((0, 4), dtype=np.float64),
            np.array([], dtype=np.float64),
            np.array([], dtype=np.int64),
        )

    result_boxes = np.concatenate(all_boxes, axis=0)
    result_scores = np.concatenate(all_scores, axis=0)
    result_labels = np.concatenate(all_labels, axis=0)

    # Sort by score descending for consistent output
    order = np.argsort(-result_scores)
    return result_boxes[order], result_scores[order], result_labels[order]


def class_agnostic_soft_nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    sigma: float = 0.5,
    score_threshold: float = 0.01,
    method: str = "gaussian",
) -> tuple[np.ndarray, np.ndarray]:
    """Soft-NMS without per-class separation — for detection-only scoring.

    All boxes compete against each other regardless of class. Useful when
    only detection mAP matters (not classification).

    Args:
        boxes: (N, 4) in [x1, y1, x2, y2] format.
        scores: (N,) confidence scores.
        sigma: Gaussian decay parameter.
        score_threshold: Discard detections below this after decay.
        method: "gaussian" or "linear".

    Returns:
        Tuple of (filtered_boxes, filtered_scores).
    """
    if len(boxes) == 0:
        return np.zeros((0, 4), dtype=np.float64), np.array([], dtype=np.float64)

    boxes = np.asarray(boxes, dtype=np.float64)
    scores = np.asarray(scores, dtype=np.float64)

    keep_idx, keep_scores = _soft_nms_single_class(
        boxes,
        scores,
        sigma=sigma,
        score_threshold=score_threshold,
        method=method,
    )

    if len(keep_idx) == 0:
        return np.zeros((0, 4), dtype=np.float64), np.array([], dtype=np.float64)

    result_boxes = boxes[keep_idx]

    # Sort by score descending
    order = np.argsort(-keep_scores)
    return result_boxes[order], keep_scores[order]
