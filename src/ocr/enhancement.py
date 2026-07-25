"""Image Enhancement Preprocessing Module for License Plate OCR.

Applies:
Plate Crop -> Grayscale -> Bilateral Filter (Denoise) -> CLAHE Contrast -> 4x Enlargement (cv2.INTER_CUBIC)
to prepare license plate character images for OCR extraction.
"""

import logging
import cv2
import numpy as np

logger = logging.getLogger("ALPR.Enhancement")


def enhance_crop_for_ocr(
    crop: np.ndarray,
    scale: float = 4.0,
) -> np.ndarray:
    """Enhance license plate crop prior to OCR recognition.

    Pipeline:
    1. Grayscale Conversion
    2. Bilateral Filter Denoising (edge-preserving noise reduction)
    3. CLAHE Contrast Enhancement
    4. 4x Upscaling (Enlargement) using cv2.INTER_CUBIC

    Args:
        crop: BGR image crop of detected license plate.
        scale: Enlargement scale factor (default 4.0x).

    Returns:
        Enhanced BGR 3-channel image ready for OCR engine input.
    """
    if crop is None or crop.size == 0:
        return crop

    # 1. Grayscale Conversion
    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop.copy()

    # 2. Bilateral Filter Denoising (preserves sharp character edges)
    gray_denoised = cv2.bilateralFilter(
        gray, d=9, sigmaColor=75.0, sigmaSpace=75.0
    )

    # 3. CLAHE Contrast Enhancement
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray_clahe = clahe.apply(gray_denoised)

    # 4. 4x Enlargement using cv2.INTER_CUBIC
    if scale > 1.0:
        h, w = gray_clahe.shape[:2]
        new_h, new_w = int(h * scale), int(w * scale)
        gray_enlarged = cv2.resize(
            gray_clahe, (new_w, new_h), interpolation=cv2.INTER_CUBIC
        )
    else:
        gray_enlarged = gray_clahe

    # Convert back to 3-channel BGR for PaddleOCR input compatibility
    enhanced_bgr = cv2.cvtColor(gray_enlarged, cv2.COLOR_GRAY2BGR)
    return enhanced_bgr
