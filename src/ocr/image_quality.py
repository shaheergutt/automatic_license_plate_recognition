"""Image Quality Evaluation Module for ALPR License Plate Crops.

Evaluates candidate crop quality based on Sharpness (Laplacian variance), Resolution,
Brightness, and Contrast metrics to prioritize the highest-quality crops for OCR recognition.
"""

import logging
from dataclasses import dataclass
from typing import Tuple

import cv2
import numpy as np

logger = logging.getLogger("ALPR.ImageQuality")


@dataclass
class QualityScore:
    """Dataclass storing multi-metric quality evaluation results for a crop."""

    sharpness: float = 0.0
    resolution: int = 0
    brightness: float = 0.0
    contrast: float = 0.0
    overall_score: float = 0.0
    is_sharp: bool = False


def estimate_sharpness(image: np.ndarray) -> float:
    """Calculate the Laplacian variance of an image to measure focus/sharpness.

    Higher values indicate sharper images; lower values indicate blur.

    Args:
        image: BGR or Grayscale image crop.

    Returns:
        Sharpness score (Laplacian variance value).
    """
    if image is None or image.size == 0:
        return 0.0

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(variance)


def is_blurry(image: np.ndarray, blur_threshold: float = 80.0) -> Tuple[bool, float]:
    """Check whether an image crop is blurry according to threshold.

    Args:
        image: BGR or Grayscale image crop.
        blur_threshold: Sharpness threshold below which crop is considered blurry.

    Returns:
        Tuple of (is_blurry_bool, sharpness_score).
    """
    sharpness = estimate_sharpness(image)
    blurry = sharpness < blur_threshold
    return blurry, sharpness


def evaluate_crop_quality(
    image: np.ndarray, blur_threshold: float = 80.0
) -> QualityScore:
    """Evaluate comprehensive multi-metric quality score for a license plate crop.

    Metrics evaluated:
    1. Sharpness: Laplacian variance score.
    2. Resolution: Total pixels (height * width).
    3. Brightness: Mean grayscale intensity score (penalizing dark <40 and bright >220).
    4. Contrast: Standard deviation of pixel intensities.

    Args:
        image: BGR image crop.
        blur_threshold: Minimum sharpness threshold required.

    Returns:
        QualityScore object containing individual metric values and overall composite score.
    """
    if image is None or image.size == 0 or image.shape[0] < 6 or image.shape[1] < 12:
        return QualityScore(overall_score=0.0, is_sharp=False)

    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    h, w = gray.shape[:2]
    resolution = h * w

    # 1. Sharpness Score (Laplacian variance)
    sharpness = estimate_sharpness(gray)
    is_sharp = sharpness >= blur_threshold

    # 2. Brightness Metric (Ideal mean ~ 128.0)
    mean_brightness = float(np.mean(gray))
    # Gaussian penalty centered around 128.0 with std_dev 60.0
    brightness_score = np.exp(-((mean_brightness - 128.0) ** 2) / (2 * (60.0 ** 2)))

    # 3. Contrast Metric (Standard deviation of intensities)
    std_contrast = float(np.std(gray))
    contrast_score = min(1.0, std_contrast / 64.0)

    # 4. Normalized Sharpness Score (log scaling to prevent outlier dominates)
    norm_sharpness = min(1.0, np.log1p(sharpness) / np.log1p(500.0))

    # 5. Normalized Resolution Score (ideal crop area >= 6000 px e.g. 150x40)
    norm_res = min(1.0, resolution / 6000.0)

    # Composite Overall Score: weighted combination
    overall = (
        (0.45 * norm_sharpness)
        + (0.25 * norm_res)
        + (0.15 * contrast_score)
        + (0.15 * brightness_score)
    )

    return QualityScore(
        sharpness=sharpness,
        resolution=resolution,
        brightness=mean_brightness,
        contrast=std_contrast,
        overall_score=float(overall),
        is_sharp=is_sharp,
    )
