"""Lightweight Centroid Tracker for High-Performance CPU License Plate Tracking.

Minimizes CPU overhead by performing Euclidean distance matching between bounding box centroids.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

from src.detection.plate_detector import DetectionBox


@dataclass
class CentroidTrack:
    """Represents an active track managed by CentroidTracker."""

    track_id: int
    centroid: Tuple[float, float]
    box: Tuple[float, float, float, float]  # (x1, y1, x2, y2)
    confidence: float
    disappeared_frames: int = 0
    hit_count: int = 1


class CentroidTrackerEngine:
    """Lightweight Centroid Tracker engine."""

    def __init__(self, max_distance: float = 120.0, max_disappeared: int = 30) -> None:
        """Initialize tracker with distance threshold and max disappeared frames.

        Args:
            max_distance: Max Euclidean distance in pixels to associate centroids.
            max_disappeared: Max consecutive frames a track can be missing before deletion.
        """
        self.next_object_id = 1
        self.tracks: Dict[int, CentroidTrack] = {}
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

    @staticmethod
    def _calc_centroid(box: Tuple[float, float, float, float]) -> Tuple[float, float]:
        """Calculate centroid (cx, cy) of a bounding box."""
        x1, y1, x2, y2 = box
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)

    def update(self, detections: List[DetectionBox]) -> List[Tuple[int, Tuple[float, float, float, float], float, int]]:
        """Update tracker with frame detections.

        Args:
            detections: List of DetectionBox objects.

        Returns:
            List of tuples: (track_id, (x1, y1, x2, y2), confidence, hit_count)
        """
        if not detections:
            # Mark all existing tracks as disappeared
            to_delete = []
            for track_id, track in self.tracks.items():
                track.disappeared_frames += 1
                if track.disappeared_frames > self.max_disappeared:
                    to_delete.append(track_id)
            for track_id in to_delete:
                del self.tracks[track_id]
            return []

        input_centroids = [self._calc_centroid(d.box_xyxy) for d in detections]
        input_boxes = [d.box_xyxy for d in detections]
        input_confs = [d.confidence for d in detections]

        # If no active tracks, register all detections as new tracks
        if not self.tracks:
            for i in range(len(detections)):
                self._register(input_centroids[i], input_boxes[i], input_confs[i])
        else:
            track_ids = list(self.tracks.keys())
            existing_centroids = [self.tracks[tid].centroid for tid in track_ids]

            # Calculate pairwise Euclidean distance matrix
            D = np.zeros((len(existing_centroids), len(input_centroids)), dtype=np.float32)
            for i, c_exist in enumerate(existing_centroids):
                for j, c_in in enumerate(input_centroids):
                    D[i, j] = math.hypot(c_exist[0] - c_in[0], c_exist[1] - c_in[1])

            # Find minimum distance assignments
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for row, col in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue

                dist = D[row, col]
                if dist > self.max_distance:
                    continue

                track_id = track_ids[row]
                self.tracks[track_id].centroid = input_centroids[col]
                self.tracks[track_id].box = input_boxes[col]
                self.tracks[track_id].confidence = input_confs[col]
                self.tracks[track_id].disappeared_frames = 0
                self.tracks[track_id].hit_count += 1

                used_rows.add(row)
                used_cols.add(col)

            # Handle unassigned existing tracks
            unused_rows = set(range(len(existing_centroids))) - used_rows
            to_delete = []
            for row in unused_rows:
                track_id = track_ids[row]
                self.tracks[track_id].disappeared_frames += 1
                if self.tracks[track_id].disappeared_frames > self.max_disappeared:
                    to_delete.append(track_id)
            for track_id in to_delete:
                del self.tracks[track_id]

            # Handle unassigned new detections
            unused_cols = set(range(len(input_centroids))) - used_cols
            for col in unused_cols:
                self._register(input_centroids[col], input_boxes[col], input_confs[col])

        # Prepare return list
        active_results = []
        for track_id, track in self.tracks.items():
            if track.disappeared_frames == 0:
                active_results.append((track_id, track.box, track.confidence, track.hit_count))

        return active_results

    def _register(self, centroid: Tuple[float, float], box: Tuple[float, float, float, float], conf: float) -> None:
        """Register new object track."""
        self.tracks[self.next_object_id] = CentroidTrack(
            track_id=self.next_object_id,
            centroid=centroid,
            box=box,
            confidence=conf,
            disappeared_frames=0,
            hit_count=1,
        )
        self.next_object_id += 1
