"""Video I/O Utility Module for OpenCV Video Handling.

Provides robust reading, metadata extraction, and writing for output videos.
"""

import logging
from pathlib import Path
from typing import Generator, Optional, Tuple, Union

import cv2
import numpy as np

logger = logging.getLogger("ALPR.Video")


class VideoHandler:
    """Context manager for OpenCV video capture and video writer."""

    def __init__(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        fourcc_str: str = "mp4v",
    ) -> None:
        """Initialize VideoHandler.

        Args:
            input_path: Path to input video file (.mp4, .avi, .mov, .mkv).
            output_path: Destination path for output video file.
            fourcc_str: FourCC encoding codec string ('mp4v').
        """
        self.input_path = Path(input_path)
        self.output_path = Path(output_path) if output_path else None
        self.fourcc_str = fourcc_str

        self.cap: Optional[cv2.VideoCapture] = None
        self.writer: Optional[cv2.VideoWriter] = None

        self.width = 0
        self.height = 0
        self.fps = 0.0
        self.total_frames = 0

    def __enter__(self) -> "VideoHandler":
        """Open video stream and output writer."""
        if not self.input_path.exists():
            raise FileNotFoundError(f"Input video file not found: {self.input_path}")

        self.cap = cv2.VideoCapture(str(self.input_path))
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video file: {self.input_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = float(self.cap.get(cv2.CAP_PROP_FPS))
        if self.fps <= 0.0 or np.isnan(self.fps):
            self.fps = 25.0

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

        logger.info(
            "Opened video '%s' (%dx%d @ %.1f FPS, Total Frames: %d)",
            self.input_path.name,
            self.width,
            self.height,
            self.fps,
            self.total_frames,
        )

        if self.output_path:
            self.output_path.parent.mkdir(parents=True, exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*self.fourcc_str)
            self.writer = cv2.VideoWriter(
                str(self.output_path),
                fourcc,
                self.fps,
                (self.width, self.height),
            )
            logger.info("Output video writer initialized: %s", self.output_path)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Release OpenCV video capture and writer resources."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        logger.info("Video resources released.")

    def read_frames(self) -> Generator[Tuple[int, np.ndarray], None, None]:
        """Generator yielding sequential (frame_idx, frame) tuples.

        Yields:
            Tuple of (1-based frame_idx, BGR frame ndarray).
        """
        if self.cap is None:
            raise RuntimeError("VideoCapture stream is not open.")

        frame_idx = 0
        while self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret or frame is None:
                break
            frame_idx += 1
            yield frame_idx, frame

    def write_frame(self, frame: np.ndarray) -> None:
        """Write frame to output video file.

        Args:
            frame: BGR image frame array.
        """
        if self.writer is not None and frame is not None and frame.size > 0:
            self.writer.write(frame)
