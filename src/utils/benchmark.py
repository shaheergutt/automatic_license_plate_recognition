"""Benchmark Performance Reporting Module for ALPR v2.

Measures processing throughput, latency metrics, CPU & memory utilization.
"""

import logging
import time
from dataclasses import dataclass
from typing import Dict, Optional

import psutil

logger = logging.getLogger("ALPR.Benchmark")


@dataclass
class BenchmarkMetrics:
    """Dataclass holding benchmark results."""

    total_frames: int = 0
    processing_time: float = 0.0
    avg_fps: float = 0.0
    avg_detection_time_ms: float = 0.0
    avg_ocr_time_ms: float = 0.0
    ocr_calls: int = 0
    cpu_percent: float = 0.0
    memory_mb: float = 0.0
    unique_plates: int = 0


class BenchmarkTracker:
    """Benchmark tracker monitoring execution timers and system hardware resources."""

    def __init__(self) -> None:
        """Initialize timing accumulators and process resource monitor."""
        self.process = psutil.Process()
        self.start_time = time.perf_counter()
        self.total_frames = 0
        self.detection_time_sum = 0.0
        self.ocr_time_sum = 0.0
        self.ocr_calls = 0

    def record_detection(self, duration_seconds: float) -> None:
        """Record detection step execution latency."""
        self.detection_time_sum += duration_seconds

    def record_ocr(self, duration_seconds: float) -> None:
        """Record OCR step execution latency."""
        self.ocr_time_sum += duration_seconds
        self.ocr_calls += 1

    def increment_frame(self) -> None:
        """Increment processed frame counter."""
        self.total_frames += 1

    def get_metrics(self, unique_plates_count: int = 0) -> BenchmarkMetrics:
        """Calculate and return benchmark summary metrics.

        Args:
            unique_plates_count: Total unique plate count.

        Returns:
            BenchmarkMetrics object.
        """
        elapsed = time.perf_counter() - self.start_time
        avg_fps = self.total_frames / elapsed if elapsed > 0 else 0.0
        avg_det_ms = (self.detection_time_sum / self.total_frames * 1000.0) if self.total_frames > 0 else 0.0
        avg_ocr_ms = (self.ocr_time_sum / self.ocr_calls * 1000.0) if self.ocr_calls > 0 else 0.0

        # Memory and CPU metrics
        cpu_pct = psutil.cpu_percent(interval=None)
        mem_mb = self.process.memory_info().rss / (1024 * 1024)

        return BenchmarkMetrics(
            total_frames=self.total_frames,
            processing_time=elapsed,
            avg_fps=avg_fps,
            avg_detection_time_ms=avg_det_ms,
            avg_ocr_time_ms=avg_ocr_ms,
            ocr_calls=self.ocr_calls,
            cpu_percent=cpu_pct,
            memory_mb=mem_mb,
            unique_plates=unique_plates_count,
        )

    def print_benchmark_report(self, unique_plates_count: int = 0) -> BenchmarkMetrics:
        """Print formatted benchmark summary report to console and log file."""
        m = self.get_metrics(unique_plates_count=unique_plates_count)

        report_lines = [
            "\n" + "=" * 60,
            " HIGH-PERFORMANCE CPU ALPR BENCHMARK REPORT",
            "=" * 60,
            f" Total Frames Processed  : {m.total_frames}",
            f" Processing Duration     : {m.processing_time:.2f} seconds",
            f" Average Throughput      : {m.avg_fps:.2f} FPS",
            f" Average Detection Time  : {m.avg_detection_time_ms:.2f} ms/frame",
            f" Average OCR Time        : {m.avg_ocr_time_ms:.2f} ms/call ({m.ocr_calls} calls)",
            f" System CPU Utilization  : {m.cpu_percent:.1f}%",
            f" Process Memory Usage    : {m.memory_mb:.2f} MB",
            f" Unique Plates Detected  : {m.unique_plates}",
            "=" * 60,
        ]

        report_str = "\n".join(report_lines)
        print(report_str)
        logger.info(report_str)
        return m
