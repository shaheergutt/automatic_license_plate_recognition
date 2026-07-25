"""OCR Results Aggregation Module using Multi-Frame Majority Voting.

Aggregates multiple OCR observations collected across video frames for a tracked vehicle,
computing candidate frequencies, weighted confidence, and crop quality metrics to determine
the single best license plate text output.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from src.ocr.validator import validate_plate_text

logger = logging.getLogger("ALPR.Aggregator")


@dataclass
class OCRObservation:
    """Dataclass storing a single OCR prediction instance from a video frame."""

    text: str
    confidence: float
    crop_quality: float
    frame_idx: int
    engine_name: str = "PaddleOCR"


def aggregate_ocr_results(
    observations: List[OCRObservation],
    min_observations: int = 1,
    confidence_threshold: float = 0.75,
) -> Tuple[str, float]:
    """Aggregate multi-frame OCR predictions using weighted majority voting.

    Ranking formula for candidate text:
        Score = Frequency * (Mean Confidence ** 1.5) * (Mean Quality Score ** 0.5)

    Args:
        observations: List of OCRObservation items collected for a track.
        min_observations: Minimum observations required before making non-UNKNOWN decision.
        confidence_threshold: Minimum confidence threshold.

    Returns:
        Tuple of (best_plate_text, best_confidence_score).
    """
    if not observations:
        return "UNKNOWN", 0.0

    # Filter valid observations meeting confidence threshold and regex format rules
    valid_obs = [
        obs
        for obs in observations
        if obs.text != "UNKNOWN"
        and obs.confidence >= confidence_threshold
        and validate_plate_text(obs.text)
    ]

    if not valid_obs:
        # If no obs met threshold, return top valid obs if any or UNKNOWN
        unthresholded = [
            obs for obs in observations if obs.text != "UNKNOWN" and validate_plate_text(obs.text)
        ]
        if not unthresholded:
            return "UNKNOWN", 0.0
        # Check if top confidence unthresholded meets fallback rules
        best_unthresh = max(unthresholded, key=lambda x: x.confidence)
        if len(unthresholded) >= min_observations and best_unthresh.confidence >= 0.70:
            return best_unthresh.text, best_unthresh.confidence
        return "UNKNOWN", 0.0

    # Group observations by candidate text
    candidates: Dict[str, List[OCRObservation]] = defaultdict(list)
    for obs in valid_obs:
        candidates[obs.text].append(obs)

    best_text = "UNKNOWN"
    highest_score = -1.0
    winning_confidence = 0.0

    for candidate_text, obs_list in candidates.items():
        freq = len(obs_list)
        avg_conf = sum(o.confidence for o in obs_list) / freq
        avg_qual = sum(o.crop_quality for o in obs_list) / freq

        # Score calculation incorporating frequency, confidence, and image quality
        candidate_score = freq * (avg_conf ** 1.5) * (avg_qual ** 0.5)

        if candidate_score > highest_score:
            highest_score = candidate_score
            best_text = candidate_text
            winning_confidence = avg_conf

    logger.debug(
        "Aggregation choice: '%s' (Conf: %.1f%%, Observations: %d/%d)",
        best_text,
        winning_confidence * 100,
        len(candidates[best_text]),
        len(observations),
    )

    return best_text, winning_confidence
