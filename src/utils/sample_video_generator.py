"""Sample Synthetic Video Generator for Pipeline Verification.

Generates a test video containing synthetic vehicle license plates for pipeline testing.
"""

import logging
from pathlib import Path
from typing import Union

import cv2
import numpy as np

from config import ALPRConfig, default_config

logger = logging.getLogger("ALPR.SampleVideo")


def generate_sample_video(
    output_path: Union[str, Path],
    num_frames: int = 60,
    fps: int = 30,
    width: int = 1280,
    height: int = 720,
) -> Path:
    """Generate a synthetic video with moving license plate cards.

    Args:
        output_path: Path where output MP4 file will be saved.
        num_frames: Total number of frames to generate.
        fps: Frame rate.
        width: Video width.
        height: Video height.

    Returns:
        Path to generated video file.
    """
    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_p), fourcc, fps, (width, height))

    if not writer.isOpened():
        raise RuntimeError(f"Could not open VideoWriter for {out_p}")

    logger.info("Generating synthetic verification video at: %s", out_p)

    # Plate details to render
    plates_info = [
        {"text": "LEA-1234", "start_x": 100, "start_y": 300, "dx": 8, "dy": 0, "w": 220, "h": 70},
        {"text": "ABC-5678", "start_x": 800, "start_y": 450, "dx": -6, "dy": 0, "w": 220, "h": 70},
    ]

    for frame_idx in range(num_frames):
        # Create dark background frame representing a road
        frame = np.full((height, width, 3), (50, 50, 50), dtype=np.uint8)

        # Draw road lanes
        cv2.line(frame, (0, 400), (width, 400), (255, 255, 255), 2)

        for p in plates_info:
            x = p["start_x"] + p["dx"] * frame_idx
            y = p["start_y"] + p["dy"] * frame_idx
            w, h = p["w"], p["h"]

            # Ensure inside frame bounds
            if 0 <= x < width - w and 0 <= y < height - h:
                # Draw vehicle body (dark grey rectangle)
                cv2.rectangle(frame, (x - 40, y - 60), (x + w + 40, y + h + 20), (30, 30, 30), -1)

                # Draw license plate background (white plate with black border)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (240, 240, 240), -1)
                cv2.rectangle(frame, (x, y), (x + w, y + h), (10, 10, 10), 3)

                # Draw license plate text
                cv2.putText(
                    frame,
                    p["text"],
                    (x + 15, y + 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.2,
                    (10, 10, 10),
                    3,
                    cv2.LINE_AA,
                )

        writer.write(frame)

    writer.release()
    logger.info("Sample verification video generated successfully (%d frames).", num_frames)
    return out_p


if __name__ == "__main__":
    generate_sample_video(default_config.default_input_video)
