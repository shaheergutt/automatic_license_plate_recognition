"""Adaptive Frame Skipping Module for CPU Optimization.

Dynamically adjusts frame skip interval based on per-frame processing latency
to maintain smooth real-time video playback.
"""

import logging
import time

from config import ALPRConfig, default_config

logger = logging.getLogger("ALPR.AdaptiveSkip")


class AdaptiveFrameSkipper:
    """Dynamically adjusts frame skip rate based on CPU processing speed."""

    def __init__(self, config: ALPRConfig = default_config) -> None:
        """Initialize frame skipper parameters.

        Args:
            config: ALPRConfig instance.
        """
        self.enabled = config.adaptive_skip
        self.current_skip = config.frame_skip_interval
        self.min_skip = config.min_skip_interval
        self.max_skip = config.max_skip_interval
        self.target_fps = config.target_fps
        self.target_frame_time = 1.0 / self.target_fps if self.target_fps > 0 else 0.04

        self._frame_times = []
        self._window_size = 10
        self._last_time = time.perf_counter()

    def should_process_frame(self, frame_index: int) -> bool:
        """Determine whether the current frame index should be processed or skipped.

        Args:
            frame_index: Sequential 1-based frame index.

        Returns:
            True if frame should be processed by detector; False to skip.
        """
        if not self.enabled:
            return (frame_index % self.current_skip) == 0

        return (frame_index % self.current_skip) == 0

    def record_frame_latency(self, latency_seconds: float) -> None:
        """Record the processing duration of a processed frame and adjust skip rate.

        Args:
            latency_seconds: Duration in seconds taken to process frame.
        """
        if not self.enabled:
            return

        self._frame_times.append(latency_seconds)
        if len(self._frame_times) > self._window_size:
            self._frame_times.pop(0)

        avg_latency = sum(self._frame_times) / len(self._frame_times)
        effective_fps = 1.0 / avg_latency if avg_latency > 0 else 30.0

        # Adjust skip interval
        if effective_fps < (self.target_fps * 0.7) and self.current_skip < self.max_skip:
            self.current_skip += 1
            logger.info("CPU load high (FPS: %.1f). Increasing frame skip -> %d", effective_fps, self.current_skip)
        elif effective_fps > (self.target_fps * 1.2) and self.current_skip > self.min_skip:
            self.current_skip -= 1
            logger.info("CPU load low (FPS: %.1f). Decreasing frame skip -> %d", effective_fps, self.current_skip)
