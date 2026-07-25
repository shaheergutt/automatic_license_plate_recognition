"""SORT (Simple Online and Realtime Tracking) implementation for ALPR.

Uses bounding box IoU association and simple motion tracking.
"""

from typing import List, Tuple

import numpy as np

from src.detection.plate_detector import DetectionBox


def iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
    """Compute Intersection over Union (IoU) between two boxes (x1, y1, x2, y2)."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area1 = max(0.0, box1[2] - box1[0]) * max(0.0, box1[3] - box1[1])
    area2 = max(0.0, box2[2] - box2[0]) * max(0.0, box2[3] - box2[1])
    union = area1 + area2 - inter

    return inter / union if union > 0 else 0.0


class KalmanBoxTracker:
    """Minimal bounding box tracker with linear velocity estimation."""

    count = 0

    def __init__(self, box: Tuple[float, float, float, float], conf: float) -> None:
        """Initialize box state [x1, y1, x2, y2]."""
        KalmanBoxTracker.count += 1
        self.id = KalmanBoxTracker.count
        self.box = box
        self.confidence = conf
        self.time_since_update = 0
        self.hit_streak = 1
        self.hit_count = 1

    def update(self, box: Tuple[float, float, float, float], conf: float) -> None:
        """Update box coordinates and confidence."""
        self.time_since_update = 0
        self.hit_streak += 1
        self.hit_count += 1
        self.box = box
        self.confidence = conf

    def predict(self) -> Tuple[float, float, float, float]:
        """Predict next position."""
        self.time_since_update += 1
        return self.box


class SortTrackerEngine:
    """SORT tracking engine using IoU association."""

    def __init__(self, max_age: int = 15, min_hits: int = 1, iou_threshold: float = 0.3) -> None:
        """Initialize SORT tracker settings."""
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.trackers: List[KalmanBoxTracker] = []

    def update(self, detections: List[DetectionBox]) -> List[Tuple[int, Tuple[float, float, float, float], float, int]]:
        """Update tracker state with new frame detections.

        Returns:
            List of tuples: (track_id, (x1, y1, x2, y2), confidence, hit_count)
        """
        # 1. Predict existing trackers
        for trk in self.trackers:
            trk.predict()

        if not detections:
            # Purge expired trackers
            self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]
            return []

        det_boxes = [d.box_xyxy for d in detections]
        det_confs = [d.confidence for d in detections]

        # 2. Compute IoU matrix
        iou_matrix = np.zeros((len(self.trackers), len(det_boxes)), dtype=np.float32)
        for t_idx, trk in enumerate(self.trackers):
            for d_idx, d_box in enumerate(det_boxes):
                iou_matrix[t_idx, d_idx] = iou(trk.box, d_box)

        # 3. Match using greedy IoU selection
        matched_trks = set()
        matched_dets = set()

        if iou_matrix.size > 0:
            flat_indices = np.argsort(-iou_matrix.ravel())
            for idx in flat_indices:
                t_idx, d_idx = divmod(int(idx), len(det_boxes))
                if t_idx in matched_trks or d_idx in matched_dets:
                    continue
                if iou_matrix[t_idx, d_idx] >= self.iou_threshold:
                    self.trackers[t_idx].update(det_boxes[d_idx], det_confs[d_idx])
                    matched_trks.add(t_idx)
                    matched_dets.add(d_idx)

        # 4. Create new trackers for unmatched detections
        for d_idx in range(len(det_boxes)):
            if d_idx not in matched_dets:
                self.trackers.append(KalmanBoxTracker(det_boxes[d_idx], det_confs[d_idx]))

        # 5. Remove dead trackers
        self.trackers = [t for t in self.trackers if t.time_since_update <= self.max_age]

        # 6. Gather active results
        results = []
        for trk in self.trackers:
            if trk.time_since_update == 0 and trk.hit_streak >= self.min_hits:
                results.append((trk.id, trk.box, trk.confidence, trk.hit_count))

        return results
