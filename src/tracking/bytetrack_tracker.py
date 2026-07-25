"""ByteTrack integration wrapper using supervision library."""

from typing import List, Tuple

import numpy as np
import supervision as sv

from src.detection.plate_detector import DetectionBox


class ByteTrackEngine:
    """ByteTrack wrapper class."""

    def __init__(self) -> None:
        """Initialize Supervision ByteTrack tracker."""
        self.bytetrack = sv.ByteTrack()

    def update(self, detections: List[DetectionBox]) -> List[Tuple[int, Tuple[float, float, float, float], float, int]]:
        """Update ByteTrack with detections.

        Returns:
            List of tuples: (track_id, (x1, y1, x2, y2), confidence, hit_count)
        """
        if not detections:
            empty_sv = sv.Detections.empty()
            self.bytetrack.update_with_detections(empty_sv)
            return []

        xyxy_arr = np.array([d.box_xyxy for d in detections], dtype=np.float32)
        conf_arr = np.array([d.confidence for d in detections], dtype=np.float32)
        class_id_arr = np.zeros(len(detections), dtype=int)

        sv_dets = sv.Detections(
            xyxy=xyxy_arr,
            confidence=conf_arr,
            class_id=class_id_arr,
        )

        tracked_sv = self.bytetrack.update_with_detections(sv_dets)
        results = []

        if len(tracked_sv) > 0:
            for i in range(len(tracked_sv)):
                box = tracked_sv.xyxy[i]
                conf = float(tracked_sv.confidence[i]) if tracked_sv.confidence is not None else 1.0
                track_id = int(tracked_sv.tracker_id[i]) if tracked_sv.tracker_id is not None else i + 1
                x1, y1, x2, y2 = float(box[0]), float(box[1]), float(box[2]), float(box[3])
                results.append((track_id, (x1, y1, x2, y2), conf, 1))

        return results
