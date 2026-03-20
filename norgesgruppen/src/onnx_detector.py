"""ONNX detector wrapper — mimics ultralytics YOLO interface.

Lets YOLO26/YOLO11 models exported to ONNX work seamlessly with
our existing SAHI, ensemble, Soft-NMS, and WBF code.

No `import os` — uses pathlib only. Sandbox-safe.
Requires: onnxruntime-gpu==1.20.0 (pre-installed in sandbox).
"""

import numpy as np
import cv2
import torch
import onnxruntime as ort


class _Boxes:
    """Mimics ultralytics Results.boxes with .xyxy, .conf, .cls attributes."""

    def __init__(self, xyxy: np.ndarray, conf: np.ndarray, cls: np.ndarray):
        self._xyxy = torch.from_numpy(xyxy).float()
        self._conf = torch.from_numpy(conf).float()
        self._cls = torch.from_numpy(cls).float()

    @property
    def xyxy(self):
        return self._xyxy

    @property
    def conf(self):
        return self._conf

    @property
    def cls(self):
        return self._cls

    def __len__(self):
        return len(self._xyxy)


class _Result:
    """Mimics a single ultralytics Results object."""

    def __init__(self, boxes: _Boxes):
        self.boxes = boxes


def _letterbox(img: np.ndarray, new_shape: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Resize image with letterbox padding (preserve aspect ratio).

    Returns:
        resized: Padded image of shape (new_shape, new_shape, 3).
        ratio: Scale factor applied.
        pad: (pad_w, pad_h) padding added.
    """
    h, w = img.shape[:2]
    ratio = new_shape / max(h, w)
    new_w, new_h = int(w * ratio), int(h * ratio)

    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    pad_w = (new_shape - new_w) // 2
    pad_h = (new_shape - new_h) // 2

    padded = np.full((new_shape, new_shape, 3), 114, dtype=np.uint8)
    padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

    return padded, ratio, (pad_w, pad_h)


def _preprocess(img_bgr: np.ndarray, imgsz: int) -> tuple[np.ndarray, float, tuple[int, int]]:
    """Preprocess BGR image for YOLO ONNX model.

    Returns:
        tensor: (1, 3, imgsz, imgsz) float32 normalized to [0, 1].
        ratio: Scale factor.
        pad: (pad_w, pad_h) padding.
    """
    padded, ratio, pad = _letterbox(img_bgr, imgsz)

    # BGR -> RGB, HWC -> CHW, normalize
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1).astype(np.float32) / 255.0
    tensor = np.expand_dims(chw, axis=0)

    return tensor, ratio, pad


def _parse_output(
    raw: np.ndarray,
    conf_threshold: float,
    orig_h: int,
    orig_w: int,
    ratio: float,
    pad: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse YOLO ONNX output into boxes, scores, class_ids.

    Handles two common output layouts:
      - (1, N, 6): each row is [x1, y1, x2, y2, conf, class_id]
      - (1, 4+nc, N): transposed format from ultralytics export
        where rows 0-3 are cx, cy, w, h and rows 4+ are class scores
      - (1, 6, N): transposed [x1, y1, x2, y2, conf, class_id]
    """
    if raw.ndim == 3:
        raw = raw[0]  # Remove batch dim -> (N, 6) or (4+nc, N) or (6, N)

    rows, cols = raw.shape

    # Heuristic: only transpose when first dim looks like features (small) and second like detections (large)
    if rows <= 360 and cols > rows * 2:
        raw = raw.T  # Now (N, features)
        rows, cols = raw.shape

    # Determine format based on number of columns
    if cols == 6:
        # Format: [x1, y1, x2, y2, conf, class_id]
        mask = raw[:, 4] >= conf_threshold
        filtered = raw[mask]

        if len(filtered) == 0:
            return np.zeros((0, 4)), np.array([]), np.array([], dtype=int)

        boxes = filtered[:, :4]
        scores = filtered[:, 4]
        class_ids = filtered[:, 5].astype(int)

    elif cols > 6:
        # Ultralytics export format: [cx, cy, w, h, cls0_score, cls1_score, ...]
        # or: [x1, y1, x2, y2, cls0_score, cls1_score, ...]
        class_scores = raw[:, 4:]
        max_scores = class_scores.max(axis=1)
        mask = max_scores >= conf_threshold
        filtered = raw[mask]
        max_scores = max_scores[mask]
        class_ids = class_scores[mask].argmax(axis=1).astype(int)

        if len(filtered) == 0:
            return np.zeros((0, 4)), np.array([]), np.array([], dtype=int)

        # Check if cx/cy/w/h or x1/y1/x2/y2
        # Heuristic: if col2 > col0 and col3 > col1 consistently, it's xyxy
        sample = filtered[:min(10, len(filtered))]
        if np.all(sample[:, 2] > sample[:, 0]) and np.all(sample[:, 3] > sample[:, 1]):
            # Already x1, y1, x2, y2
            boxes = filtered[:, :4]
        else:
            # cx, cy, w, h -> x1, y1, x2, y2
            cx, cy, w, h = filtered[:, 0], filtered[:, 1], filtered[:, 2], filtered[:, 3]
            x1 = cx - w / 2
            y1 = cy - h / 2
            x2 = cx + w / 2
            y2 = cy + h / 2
            boxes = np.stack([x1, y1, x2, y2], axis=1)

        scores = max_scores

    elif cols == 5:
        # Single-class: [x1, y1, x2, y2, conf]
        mask = raw[:, 4] >= conf_threshold
        filtered = raw[mask]

        if len(filtered) == 0:
            return np.zeros((0, 4)), np.array([]), np.array([], dtype=int)

        boxes = filtered[:, :4]
        scores = filtered[:, 4]
        class_ids = np.zeros(len(filtered), dtype=int)

    else:
        return np.zeros((0, 4)), np.array([]), np.array([], dtype=int)

    # Unpad and unscale boxes from model input space to original image space
    pad_w, pad_h = pad
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_w) / ratio
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_h) / ratio

    # Clip to original image bounds
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, orig_w)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, orig_h)

    return boxes, scores, class_ids


def _nms(
    boxes: np.ndarray,
    scores: np.ndarray,
    class_ids: np.ndarray,
    iou_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply class-aware NMS using OpenCV."""
    if len(boxes) == 0:
        return boxes, scores, class_ids

    # Convert xyxy to xywh for cv2.dnn.NMSBoxes
    xywh = boxes.copy()
    xywh[:, 2] = boxes[:, 2] - boxes[:, 0]
    xywh[:, 3] = boxes[:, 3] - boxes[:, 1]

    # Per-class NMS: offset boxes by class_id to prevent cross-class suppression
    max_dim = max(boxes[:, 2].max() - boxes[:, 0].min(),
                  boxes[:, 3].max() - boxes[:, 1].min()) + 1
    offset_xywh = xywh.copy()
    offset_xywh[:, 0] += class_ids * max_dim
    offset_xywh[:, 1] += class_ids * max_dim

    indices = cv2.dnn.NMSBoxes(
        offset_xywh.tolist(),
        scores.tolist(),
        score_threshold=0.0,
        nms_threshold=iou_threshold,
    )

    if len(indices) == 0:
        return np.zeros((0, 4)), np.array([]), np.array([], dtype=int)

    idx = indices.flatten()
    return boxes[idx], scores[idx], class_ids[idx]


class ONNXDetector:
    """Wraps an ONNX model to mimic ultralytics YOLO interface.

    This lets us use YOLO26 (exported to ONNX) with our existing
    SAHI, ensemble, and Soft-NMS code without modification.
    """

    def __init__(self, model_path: str, conf_threshold: float = 0.15):
        providers = []
        if "CUDAExecutionProvider" in ort.get_available_providers():
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(model_path, providers=providers)
        self.conf_threshold = conf_threshold
        self.input_name = self.session.get_inputs()[0].name

        # Try to infer default imgsz from model input shape
        input_shape = self.session.get_inputs()[0].shape
        if isinstance(input_shape[-1], int) and input_shape[-1] > 0:
            self._default_imgsz = input_shape[-1]  # Fixed input size
            self._dynamic = False
        else:
            self._default_imgsz = 1280  # Dynamic — default to 1280 for full images
            self._dynamic = True

        print(f"[ONNX] Loaded {model_path}")
        print(f"[ONNX] Providers: {self.session.get_providers()}")
        print(f"[ONNX] Input: {self.input_name} shape={input_shape}")
        print(f"[ONNX] Default imgsz: {self._default_imgsz}")

    def _infer(
        self,
        img_bgr: np.ndarray,
        imgsz: int,
        conf: float,
        iou: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Run single inference pass and return (boxes_xyxy, scores, class_ids)."""
        orig_h, orig_w = img_bgr.shape[:2]
        # For fixed-shape models, always use native size; for dynamic, use requested size
        actual_imgsz = imgsz if self._dynamic else self._default_imgsz
        tensor, ratio, pad = _preprocess(img_bgr, actual_imgsz)

        outputs = self.session.run(None, {self.input_name: tensor})
        raw = outputs[0]

        boxes, scores, class_ids = _parse_output(
            raw, conf, orig_h, orig_w, ratio, pad,
        )

        # Apply NMS (YOLO26 is NMS-free but may still have overlapping boxes
        # from the confidence threshold; this is a safety pass)
        if len(boxes) > 0:
            boxes, scores, class_ids = _nms(boxes, scores, class_ids, iou)

        return boxes, scores, class_ids

    def __call__(
        self,
        img,
        device="cuda",
        verbose=False,
        conf=0.15,
        iou=0.45,
        imgsz=1280,
        augment=False,
    ) -> list[_Result]:
        """Match ultralytics YOLO.__call__ signature.

        Args:
            img: BGR numpy array (H, W, 3) or path string.
            device: Ignored (ONNX session uses providers set at init).
            verbose: Ignored.
            conf: Confidence threshold.
            iou: NMS IoU threshold.
            imgsz: Input resolution (square).
            augment: If True, run with horizontal flip TTA.

        Returns:
            List with one Results-like object containing .boxes attribute.
        """
        # Handle path input
        if isinstance(img, str):
            img = cv2.imread(img)
        elif isinstance(img, np.ndarray):
            pass
        else:
            # Try converting from pathlib.Path
            img = cv2.imread(str(img))

        if img is None:
            return [_Result(_Boxes(
                np.zeros((0, 4)),
                np.array([]),
                np.array([]),
            ))]

        boxes, scores, class_ids = self._infer(img, imgsz, conf, iou)

        if augment and len(img.shape) == 3:
            # Horizontal flip TTA
            flipped = cv2.flip(img, 1)
            f_boxes, f_scores, f_class_ids = self._infer(flipped, imgsz, conf, iou)

            if len(f_boxes) > 0:
                orig_w = img.shape[1]
                # Mirror boxes back
                f_boxes_mirror = f_boxes.copy()
                f_boxes_mirror[:, 0] = orig_w - f_boxes[:, 2]
                f_boxes_mirror[:, 2] = orig_w - f_boxes[:, 0]

                if len(boxes) > 0:
                    boxes = np.concatenate([boxes, f_boxes_mirror])
                    scores = np.concatenate([scores, f_scores])
                    class_ids = np.concatenate([class_ids, f_class_ids])
                else:
                    boxes = f_boxes_mirror
                    scores = f_scores
                    class_ids = f_class_ids

                # NMS on merged original + flipped detections
                boxes, scores, class_ids = _nms(boxes, scores, class_ids, iou)

        if len(boxes) == 0:
            boxes = np.zeros((0, 4))
            scores = np.array([])
            class_ids = np.array([])

        result = _Result(_Boxes(boxes, scores, class_ids))
        return [result]
