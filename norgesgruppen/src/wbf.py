"""Weighted Boxes Fusion (WBF) — pure numpy implementation.

Based on the paper: "Weighted Boxes Fusion: Ensembling boxes from different
object detection models" (Solovyev et al., 2021).

No external dependencies beyond numpy. Sandbox-safe.
"""

import numpy as np


def _bb_iou(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """IoU between two boxes [x1, y1, x2, y2] in normalized coords."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter == 0:
        return 0.0

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    return inter / (area_a + area_b - inter)


def weighted_boxes_fusion(
    boxes_list: list[np.ndarray],
    scores_list: list[np.ndarray],
    labels_list: list[np.ndarray],
    weights: list[float] | None = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fuse predictions from multiple models/scales using WBF.

    Args:
        boxes_list: List of (N_i, 4) arrays, each box as [x1, y1, x2, y2]
                    normalized to [0, 1].
        scores_list: List of (N_i,) confidence arrays.
        labels_list: List of (N_i,) integer label arrays.
        weights: Per-model weights (default: equal).
        iou_thr: IoU threshold for matching boxes into clusters.
        skip_box_thr: Skip boxes with score below this.

    Returns:
        fused_boxes: (M, 4) array of fused boxes [x1, y1, x2, y2] in [0, 1].
        fused_scores: (M,) array of fused scores.
        fused_labels: (M,) array of integer labels.
    """
    n_models = len(boxes_list)
    if weights is None:
        weights = [1.0] * n_models

    # Collect all boxes with model index, weighted score, and label
    all_boxes = []
    for model_idx in range(n_models):
        for i in range(len(boxes_list[model_idx])):
            score = scores_list[model_idx][i]
            if score < skip_box_thr:
                continue
            all_boxes.append({
                "box": boxes_list[model_idx][i].copy(),
                "score": score * weights[model_idx],
                "raw_score": score,
                "label": int(labels_list[model_idx][i]),
                "model_idx": model_idx,
            })

    if not all_boxes:
        return np.zeros((0, 4)), np.array([]), np.array([], dtype=int)

    # Sort by score descending
    all_boxes.sort(key=lambda x: -x["score"])

    # Group by label
    label_groups: dict[int, list] = {}
    for b in all_boxes:
        label_groups.setdefault(b["label"], []).append(b)

    fused_boxes = []
    fused_scores = []
    fused_labels = []

    for label, boxes in label_groups.items():
        # Clusters: each is a list of matched boxes
        clusters: list[list[dict]] = []
        cluster_boxes: list[np.ndarray] = []  # weighted avg box per cluster

        for b in boxes:
            matched = False
            best_iou = iou_thr
            best_cluster = -1

            for ci, cbox in enumerate(cluster_boxes):
                iou = _bb_iou(b["box"], cbox)
                if iou > best_iou:
                    best_iou = iou
                    best_cluster = ci
                    matched = True

            if matched:
                clusters[best_cluster].append(b)
                # Recompute weighted average box for this cluster
                cb = clusters[best_cluster]
                total_score = sum(x["score"] for x in cb)
                if total_score > 0:
                    avg_box = np.zeros(4)
                    for x in cb:
                        avg_box += x["box"] * x["score"]
                    avg_box /= total_score
                    cluster_boxes[best_cluster] = avg_box
            else:
                clusters.append([b])
                cluster_boxes.append(b["box"].copy())

        # Build fused results from clusters
        weight_sum = sum(weights)
        for ci, cluster in enumerate(clusters):
            total_score = sum(x["score"] for x in cluster)
            n_matched = len(cluster)

            # WBF score: average score weighted by number of models
            fused_score = total_score / weight_sum

            fused_boxes.append(cluster_boxes[ci])
            fused_scores.append(fused_score)
            fused_labels.append(label)

    if not fused_boxes:
        return np.zeros((0, 4)), np.array([]), np.array([], dtype=int)

    # Sort by score descending
    order = np.argsort([-s for s in fused_scores])
    fused_boxes = np.array(fused_boxes)[order]
    fused_scores = np.array(fused_scores)[order]
    fused_labels = np.array(fused_labels, dtype=int)[order]

    # Clip boxes to [0, 1]
    fused_boxes = np.clip(fused_boxes, 0, 1)

    return fused_boxes, fused_scores, fused_labels
