"""Blur Detection Module using Laplacian Variance Metric.

Used to evaluate cropped license plate image sharpness and avoid running OCR on blurry frames.
"""

import logging
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger("ALPR.BlurDetector")


def estimate_sharpness(image: np.ndarray) -> float:
    """Calculate the Laplacian variance of an image to measure focus/sharpness.

    Higher values indicate sharper images; lower values indicate blur.

    Args:
        image: BGR image crop.

    Returns:
        Sharpness score (Laplacian variance value).
    """
    if image is None or image.size == 0:
        return 0.0

    # Convert to grayscale if image is color
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Compute variance of Laplacian
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(variance)


def is_blurry(image: np.ndarray, blur_threshold: float = 80.0) -> Tuple[bool, float]:
    """Check whether an image crop is blurry according to threshold.

    Args:
        image: BGR image crop.
        blur_threshold: Threshold below which image is considered blurry.

    Returns:
        Tuple of (is_blurry_bool, sharpness_score).
    """
    sharpness = estimate_sharpness(image)
    blurry = sharpness < blur_threshold
    return blurry, sharpness
