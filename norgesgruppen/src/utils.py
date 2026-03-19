"""Pre-prosessering for butikkhyllebilder.

CLAHE (Contrast Limited Adaptive Histogram Equalization) i LAB-fargerom
for dynamisk fjerning av gjenskinn fra kjøleskapsdører og blanke overflater.
"""

import cv2
import numpy as np


def enhance_retail_image(img: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple = (8, 8)) -> np.ndarray:
    """Fjern gjenskinn og forbedre kontrast på butikkbilder.

    Kjører på ~2-3ms per bilde. Konverterer til LAB, appliserer CLAHE
    på luminans-kanalen, og konverterer tilbake.

    Args:
        img: BGR-bilde fra OpenCV (np.ndarray).
        clip_limit: CLAHE clip limit (høyere = mer kontrast).
        tile_grid: Rutenettstørrelse for CLAHE.

    Returns:
        Forbedret BGR-bilde.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_enhanced = clahe.apply(l_channel)

    lab_enhanced = cv2.merge([l_enhanced, a_channel, b_channel])
    result = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2BGR)

    return result


def preprocess_for_model(img: np.ndarray, input_size: tuple = (640, 640)) -> np.ndarray:
    """Klargjør bilde for ONNX-modellen.

    Resizer, normaliserer (0-1), og konverterer til NCHW float32-format.

    Args:
        img: BGR-bilde fra OpenCV.
        input_size: (width, height) som modellen forventer.

    Returns:
        np.ndarray med shape (1, 3, H, W), float32, normalisert.
    """
    enhanced = enhance_retail_image(img)
    resized = cv2.resize(enhanced, input_size, interpolation=cv2.INTER_LINEAR)

    # BGR -> RGB, HWC -> CHW, normalize 0-1
    blob = resized[:, :, ::-1].transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.expand_dims(blob, axis=0)
