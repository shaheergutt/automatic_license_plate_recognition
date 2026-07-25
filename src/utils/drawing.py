"""Drawing Utilities for ALPR Bounding Boxes and HUD Overlays.

Renders high-visibility overlays for detected license plates and real-time performance metrics.
"""

from typing import List, Optional

import cv2
import numpy as np

from src.tracking.tracker import TrackedPlate


def draw_hud(
    frame: np.ndarray,
    current_fps: float,
    current_frame: int,
    total_frames: int,
    elapsed_time: float,
    unique_plates_count: int,
) -> np.ndarray:
    """Draw clean real-time status HUD bar at top of video frame.

    Args:
        frame: BGR video frame image array.
        current_fps: Real-time processing frame rate.
        current_frame: Index of current video frame.
        total_frames: Total number of frames in input video (0 if unknown).
        elapsed_time: Elapsed processing duration in seconds.
        unique_plates_count: Count of unique license plates identified so far.

    Returns:
        Annotated BGR frame with HUD overlay.
    """
    h, w = frame.shape[:2]

    # Semi-transparent dark banner background at top of frame
    banner_height = 40
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, banner_height), (15, 15, 15), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)

    # Format HUD text string
    mins = int(elapsed_time // 60)
    secs = int(elapsed_time % 60)
    elapsed_str = f"{mins:02d}:{secs:02d}"

    if total_frames > 0:
        progress_pct = int((current_frame / total_frames) * 100)
        progress_str = f"Progress: {progress_pct}%"
    else:
        progress_str = f"Frame: {current_frame}"

    hud_text = (
        f"FPS: {current_fps:.1f}  |  {progress_str}  |  "
        f"Elapsed: {elapsed_str}  |  Unique Plates: {unique_plates_count}"
    )

    # Render text string
    cv2.putText(
        frame,
        hud_text,
        (15, 26),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return frame


def draw_annotations(
    frame: np.ndarray,
    tracked_plates: List[TrackedPlate],
    hud_info: Optional[dict] = None,
) -> np.ndarray:
    """Draw bounding boxes, track IDs, plate text, confidence, and optional HUD onto frame.

    Args:
        frame: BGR video frame image array.
        tracked_plates: List of TrackedPlate objects in current frame.
        hud_info: Optional dictionary containing HUD performance metrics.

    Returns:
        Annotated BGR frame array.
    """
    annotated = frame.copy()

    for plate in tracked_plates:
        x1, y1, x2, y2 = int(plate.x1), int(plate.y1), int(plate.x2), int(plate.y2)
        track_id = plate.track_id
        text = plate.plate_text
        conf = plate.ocr_confidence

        # Color coding: Green for confirmed text recognition; Cyan for UNKNOWN
        if text != "UNKNOWN":
            box_color = (0, 220, 0)      # Bright Green
            label_bg_color = (0, 160, 0)
        else:
            box_color = (255, 200, 0)    # Bright Cyan / Yellow
            label_bg_color = (200, 140, 0)

        # Draw plate bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), box_color, 2)

        # Format label text
        if text != "UNKNOWN":
            label = f"ID #{track_id} | {text} ({conf * 100:.1f}%)"
        else:
            label = f"ID #{track_id} | UNKNOWN"

        # Calculate text background box size
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2
        )
        label_y1 = max(0, y1 - text_h - 10)
        label_y2 = y1

        cv2.rectangle(
            annotated,
            (x1, label_y1),
            (x1 + text_w + 10, label_y2),
            label_bg_color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (x1 + 5, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # Draw HUD overlay if parameters supplied
    if hud_info is not None:
        annotated = draw_hud(
            frame=annotated,
            current_fps=hud_info.get("fps", 0.0),
            current_frame=hud_info.get("frame_idx", 0),
            total_frames=hud_info.get("total_frames", 0),
            elapsed_time=hud_info.get("elapsed_time", 0.0),
            unique_plates_count=hud_info.get("unique_plates", 0),
        )

    return annotated
