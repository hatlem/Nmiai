"""SAHI — Slicing Aided Hyper Inference for small object detection.

Slices large images into overlapping tiles, runs detection on each tile
plus the full image, maps coordinates back, and merges via WBF.

No `import os` — uses pathlib only (sandbox restriction).
"""

import numpy as np
import cv2

from src.wbf import weighted_boxes_fusion


def _compute_slices(
    img_size: int, slice_size: int, overlap: int
) -> list[tuple[int, int]]:
    """Compute start positions for slicing along one axis.

    Returns list of (start, end) tuples covering the full axis
    with the given overlap.
    """
    step = slice_size - overlap
    slices = []
    pos = 0
    while pos < img_size:
        end = min(pos + slice_size, img_size)
        # If the remaining strip is too small, extend back
        if end - pos < slice_size and pos > 0:
            pos = max(0, end - slice_size)
        slices.append((pos, min(pos + slice_size, img_size)))
        if end >= img_size:
            break
        pos += step
    return slices


def sahi_inference(
    model,
    img: np.ndarray,
    device: str,
    slice_size: int = 640,
    overlap_ratio: float = 0.2,
    conf_threshold: float = 0.15,
    iou_threshold: float = 0.45,
    full_image_scale: int = 1280,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run SAHI sliced inference on a single image.

    Splits the image into overlapping tiles, runs the detector on each
    tile and on the full image at `full_image_scale`, then fuses all
    detections with Weighted Boxes Fusion.

    Args:
        model: ultralytics YOLO model object.
        img: BGR numpy array (H, W, 3).
        device: "cuda" or "cpu".
        slice_size: Tile size in pixels (square tiles).
        overlap_ratio: Fraction of overlap between adjacent tiles (0-1).
        conf_threshold: Minimum confidence for YOLO inference.
        iou_threshold: NMS IoU threshold for YOLO inference.
        full_image_scale: imgsz for full-image inference pass.

    Returns:
        boxes_xyxy: (N, 4) array in pixel coordinates [x1, y1, x2, y2].
        scores: (N,) confidence scores.
        class_ids: (N,) integer class IDs.
    """
    img_h, img_w = img.shape[:2]
    overlap_px = int(slice_size * overlap_ratio)

    # --- 1. Full-image inference ---
    all_boxes_lists = []
    all_scores_lists = []
    all_labels_lists = []

    full_results = model(
        img,
        device=device,
        verbose=False,
        conf=conf_threshold,
        iou=iou_threshold,
        imgsz=full_image_scale,
    )

    for r in full_results:
        if r.boxes is not None and len(r.boxes) > 0:
            boxes = r.boxes.xyxy.cpu().numpy()
            scores = r.boxes.conf.cpu().numpy()
            labels = r.boxes.cls.cpu().numpy().astype(int)
            # Normalize to [0, 1]
            boxes[:, [0, 2]] /= img_w
            boxes[:, [1, 3]] /= img_h
            boxes = np.clip(boxes, 0, 1)
            all_boxes_lists.append(boxes)
            all_scores_lists.append(scores)
            all_labels_lists.append(labels)
        else:
            all_boxes_lists.append(np.zeros((0, 4)))
            all_scores_lists.append(np.array([]))
            all_labels_lists.append(np.array([]))

    # --- 2. Sliced inference ---
    x_slices = _compute_slices(img_w, slice_size, overlap_px)
    y_slices = _compute_slices(img_h, slice_size, overlap_px)

    for y_start, y_end in y_slices:
        for x_start, x_end in x_slices:
            tile = img[y_start:y_end, x_start:x_end]

            tile_h, tile_w = tile.shape[:2]

            # Run detector on tile
            tile_results = model(
                tile,
                device=device,
                verbose=False,
                conf=conf_threshold,
                iou=iou_threshold,
                imgsz=slice_size,
            )

            for r in tile_results:
                if r.boxes is None or len(r.boxes) == 0:
                    all_boxes_lists.append(np.zeros((0, 4)))
                    all_scores_lists.append(np.array([]))
                    all_labels_lists.append(np.array([]))
                    continue

                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                labels = r.boxes.cls.cpu().numpy().astype(int)

                # Map tile coordinates back to full image coordinates
                boxes[:, 0] += x_start
                boxes[:, 2] += x_start
                boxes[:, 1] += y_start
                boxes[:, 3] += y_start

                # Normalize to [0, 1] relative to full image
                boxes[:, [0, 2]] /= img_w
                boxes[:, [1, 3]] /= img_h
                boxes = np.clip(boxes, 0, 1)

                all_boxes_lists.append(boxes)
                all_scores_lists.append(scores)
                all_labels_lists.append(labels)

    # --- 3. Merge with WBF ---
    if not any(len(b) > 0 for b in all_boxes_lists):
        return np.zeros((0, 4)), np.array([]), np.array([], dtype=int)

    n_sources = len(all_boxes_lists)
    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        all_boxes_lists,
        all_scores_lists,
        all_labels_lists,
        weights=[1.0] * n_sources,
        iou_thr=0.6,
        skip_box_thr=0.001,
    )

    # De-normalize back to pixel coordinates
    if len(fused_boxes) > 0:
        fused_boxes[:, [0, 2]] *= img_w
        fused_boxes[:, [1, 3]] *= img_h

    return fused_boxes, fused_scores, fused_labels
