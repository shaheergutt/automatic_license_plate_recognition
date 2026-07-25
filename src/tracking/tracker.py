"""Unified Plate Tracker Module for ALPR v2.

Manages track lifecycle, tracking algorithm selection (Centroid, SORT, ByteTrack),
multi-frame quality crop candidate collection, and aggregated OCR observation state.
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from config import ALPRConfig, default_config
from src.detection.plate_detector import DetectionBox
from src.ocr.aggregator import OCRObservation, aggregate_ocr_results
from src.ocr.image_quality import evaluate_crop_quality
from src.ocr.validator import validate_plate_text
from src.tracking.bytetrack_tracker import ByteTrackEngine
from src.tracking.centroid_tracker import CentroidTrackerEngine
from src.tracking.sort_tracker import SortTrackerEngine

logger = logging.getLogger("ALPR.Tracking")


@dataclass
class CandidateCrop:
    """Dataclass holding candidate plate crop image and quality metrics."""

    crop: np.ndarray
    quality_score: float
    sharpness: float
    frame_idx: int


@dataclass
class TrackState:
    """Stores persistent state, candidate crop buffer, OCR observations, and best result memory for a tracked plate."""

    track_id: int
    best_text: str = "UNKNOWN"
    best_ocr_confidence: float = 0.0
    best_crop_area: float = 0.0
    best_sharpness: float = 0.0
    best_crop_quality: float = 0.0
    best_crop: Optional[np.ndarray] = None
    ocr_attempts: int = 0
    consecutive_hits: int = 0
    is_stable: bool = False
    candidate_crops: List[CandidateCrop] = field(default_factory=list)
    ocr_observations: List[OCRObservation] = field(default_factory=list)

    def add_candidate_crop(
        self,
        crop: np.ndarray,
        quality_score: float,
        sharpness: float,
        frame_idx: int,
        max_capacity: int = 8,
    ) -> bool:
        """Add candidate crop to buffer, maintaining top quality crops sorted by quality score.

        Returns:
            Boolean indicating whether crop was accepted into candidate buffer.
        """
        if crop is None or crop.size == 0:
            return False

        candidate = CandidateCrop(
            crop=crop.copy(),
            quality_score=quality_score,
            sharpness=sharpness,
            frame_idx=frame_idx,
        )

        self.candidate_crops.append(candidate)
        # Sort descending by quality_score
        self.candidate_crops.sort(key=lambda c: c.quality_score, reverse=True)

        if len(self.candidate_crops) > max_capacity:
            self.candidate_crops = self.candidate_crops[:max_capacity]

        return candidate in self.candidate_crops


@dataclass
class TrackedPlate:
    """Represents a tracked license plate detection in a single frame."""

    track_id: int
    x1: float
    y1: float
    x2: float
    y2: float
    detection_confidence: float
    plate_text: str
    ocr_confidence: float
    should_run_ocr: bool

    @property
    def box_xyxy(self) -> Tuple[float, float, float, float]:
        """Return box coordinates tuple (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self) -> float:
        """Calculate bounding box area in pixels."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


class PlateTracker:
    """Unified tracker factory managing persistent license plate tracks."""

    def __init__(self, config: ALPRConfig = default_config) -> None:
        """Initialize tracker engine based on config choice.

        Args:
            config: ALPRConfig instance.
        """
        self.config = config
        self.tracks: Dict[int, TrackState] = {}
        tracker_type = config.tracker_type.lower()

        if tracker_type == "sort":
            self.engine = SortTrackerEngine(
                max_age=config.sort_max_age,
                min_hits=config.sort_min_hits,
                iou_threshold=config.sort_iou_threshold,
            )
            logger.info("PlateTracker initialized using SORT Tracker.")
        elif tracker_type == "bytetrack":
            self.engine = ByteTrackEngine()
            logger.info("PlateTracker initialized using ByteTrack.")
        else:
            self.engine = CentroidTrackerEngine(
                max_distance=config.centroid_max_distance,
                max_disappeared=config.max_track_age,
            )
            logger.info("PlateTracker initialized using Centroid Tracker.")

    def update(self, detections: List[DetectionBox]) -> List[TrackedPlate]:
        """Update tracker engine and evaluate OCR trigger criteria per track.

        Args:
            detections: List of DetectionBox objects from detector.

        Returns:
            List of TrackedPlate objects.
        """
        raw_tracks = self.engine.update(detections)
        tracked_plates: List[TrackedPlate] = []

        for track_id, (x1, y1, x2, y2), conf, hit_count in raw_tracks:
            crop_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

            # Initialize track state if new track
            if track_id not in self.tracks:
                self.tracks[track_id] = TrackState(track_id=track_id)

            state = self.tracks[track_id]
            state.consecutive_hits += 1

            # Check track stability
            if state.consecutive_hits >= self.config.stable_track_frames:
                state.is_stable = True

            # Decide whether OCR should run on this frame
            should_run_ocr = False
            if state.is_stable:
                if state.best_text == "UNKNOWN":
                    should_run_ocr = True
                elif crop_area > state.best_crop_area * self.config.area_improvement_ratio:
                    should_run_ocr = True
                elif len(state.candidate_crops) < self.config.max_crops_per_track:
                    should_run_ocr = True

            tracked_plates.append(
                TrackedPlate(
                    track_id=track_id,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    detection_confidence=conf,
                    plate_text=state.best_text,
                    ocr_confidence=state.best_ocr_confidence,
                    should_run_ocr=should_run_ocr,
                )
            )

        return tracked_plates

    def add_crop_and_evaluate(
        self,
        track_id: int,
        crop: np.ndarray,
        frame_idx: int,
    ) -> Tuple[bool, float, float]:
        """Evaluate crop quality and add to candidate buffer if valid and non-blurry.

        Returns:
            Tuple of (accepted_bool, overall_quality_score, sharpness).
        """
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackState(track_id=track_id)

        state = self.tracks[track_id]
        quality = evaluate_crop_quality(crop, blur_threshold=self.config.blur_threshold)

        if not quality.is_sharp:
            return False, quality.overall_score, quality.sharpness

        accepted = state.add_candidate_crop(
            crop=crop,
            quality_score=quality.overall_score,
            sharpness=quality.sharpness,
            frame_idx=frame_idx,
            max_capacity=self.config.max_crops_per_track,
        )

        return accepted, quality.overall_score, quality.sharpness

    def record_ocr_observation(
        self,
        track_id: int,
        text: str,
        confidence: float,
        crop_quality: float,
        sharpness: float,
        frame_idx: int,
        crop: Optional[np.ndarray] = None,
        engine_name: str = "PaddleOCR",
    ) -> None:
        """Record new OCR observation and run aggregated majority voting decision.

        Args:
            track_id: Track ID.
            text: Recognized plate text.
            confidence: Confidence score.
            crop_quality: Overall image quality score.
            sharpness: Sharpness score.
            frame_idx: Frame index.
            crop: BGR image crop associated with observation.
            engine_name: Name of OCR engine used.
        """
        if track_id not in self.tracks:
            self.tracks[track_id] = TrackState(track_id=track_id)

        state = self.tracks[track_id]
        state.ocr_attempts += 1

        obs = OCRObservation(
            text=text,
            confidence=confidence,
            crop_quality=crop_quality,
            frame_idx=frame_idx,
            engine_name=engine_name,
        )
        state.ocr_observations.append(obs)

        # Run majority voting aggregation over all collected observations
        agg_text, agg_conf = aggregate_ocr_results(
            observations=state.ocr_observations,
            min_observations=self.config.min_ocr_observations,
            confidence_threshold=self.config.ocr_conf_threshold,
        )

        if agg_text != "UNKNOWN":
            if state.best_text == "UNKNOWN" or agg_conf >= state.best_ocr_confidence:
                if state.best_text != agg_text:
                    logger.info(
                        "Track %d: Updated aggregated plate text '%s' -> '%s' (Conf: %.1f%%)",
                        track_id,
                        state.best_text,
                        agg_text,
                        agg_conf * 100,
                    )
                state.best_text = agg_text
                state.best_ocr_confidence = agg_conf
                state.best_sharpness = sharpness
                state.best_crop_quality = crop_quality
                if crop is not None:
                    state.best_crop = crop.copy()

    def get_all_results(self) -> Dict[int, TrackState]:
        """Return dict of all active and past track states."""
        return self.tracks
