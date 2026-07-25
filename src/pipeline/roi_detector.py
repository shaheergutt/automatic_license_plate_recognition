"""ROI-Based Localized License Plate Detection Module.

Restricts detection search to padded bounding box regions of active tracks,
reducing CPU compute requirements. Periodically performs full-frame scans.
"""

import logging
from typing import List, Tuple

import cv2
import numpy as np

from config import ALPRConfig, default_config
from src.detection.plate_detector import DetectionBox, LicensePlateDetector
from src.tracking.tracker import TrackedPlate

logger = logging.getLogger("ALPR.ROIDetector")


class ROIDetectorManager:
    """Manager for ROI-based localized detection."""

    def __init__(
        self,
        detector: LicensePlateDetector,
        config: ALPRConfig = default_config,
    ) -> None:
        """Initialize ROI manager with plate detector and padding configuration.

        Args:
            detector: LicensePlateDetector instance.
            config: ALPRConfig instance.
        """
        self.detector = detector
        self.config = config
        self.enabled = config.use_roi_detection
        self.padding_px = config.roi_padding_px
        self.full_scan_interval = config.roi_full_scan_interval
        self.processed_counter = 0

    def detect(self, frame: np.ndarray, active_tracks: List[TrackedPlate]) -> List[DetectionBox]:
        """Detect license plates using ROI localized search or full-frame scan.

        Args:
            frame: Input BGR image.
            active_tracks: List of currently tracked plates from previous frame.

        Returns:
            List of DetectionBox objects in original frame coordinates.
        """
        if frame is None or frame.size == 0:
            return []

        self.processed_counter += 1

        # Perform full-frame scan if ROI disabled, no active tracks, or full scan interval reached
        if (
            not self.enabled
            or not active_tracks
            or (self.processed_counter % self.full_scan_interval) == 0
        ):
            return self.detector.detect(frame)

        orig_h, orig_w = frame.shape[:2]
        detections: List[DetectionBox] = []

        # Process each active track's region of interest
        for track in active_tracks:
            x1 = max(0, int(track.x1) - self.padding_px)
            y1 = max(0, int(track.y1) - self.padding_px)
            x2 = min(orig_w, int(track.x2) + self.padding_px)
            y2 = min(orig_h, int(track.y2) + self.padding_px)

            roi_w = x2 - x1
            roi_h = y2 - y1

            if roi_w < 10 or roi_h < 10:
                continue

            roi_crop = frame[y1:y2, x1:x2]
            roi_dets = self.detector.detect(roi_crop)

            # Map local ROI detection coordinates back to global frame coordinates
            for det in roi_dets:
                detections.append(
                    DetectionBox(
                        x1=float(det.x1 + x1),
                        y1=float(det.y1 + y1),
                        x2=float(det.x2 + x1),
                        y2=float(det.y2 + y1),
                        confidence=det.confidence,
                        class_id=det.class_id,
                    )
                )

        # If localized search found no plates, fallback to full-frame scan
        if not detections:
            return self.detector.detect(frame)

        return detections
