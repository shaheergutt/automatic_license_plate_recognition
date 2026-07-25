"""Centralized Configuration Module for Simple Accurate ALPR System.

Provides focused configuration settings for YOLOv8 License Plate Detection, 4x Image Enhancement,
PaddleOCR Recognition, Zoomed Plate Panel Video Rendering, and Final Results Export.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

# Base directory setup
BASE_DIR: Path = Path(__file__).resolve().parent

# Global Logging Setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(levelname)s] - %(name)s - %(message)s",
)
logger = logging.getLogger("ALPR.Config")


@dataclass
class ALPRConfig:
    """Central configuration dataclass for Simple Accurate ALPR system."""

    # Project Directories
    base_dir: Path = BASE_DIR
    models_dir: Path = field(default_factory=lambda: BASE_DIR / "models")
    input_videos_dir: Path = field(default_factory=lambda: BASE_DIR / "videos" / "input")
    output_videos_dir: Path = field(default_factory=lambda: BASE_DIR / "videos" / "output")
    logs_dir: Path = field(default_factory=lambda: BASE_DIR / "logs")

    # Target Files
    output_video_path: Path = field(
        default_factory=lambda: BASE_DIR / "videos" / "output" / "annotated_output.mp4"
    )
    results_txt_path: Path = field(default_factory=lambda: BASE_DIR / "results.txt")

    # Model Weights & Detection Parameters
    model_path: Path = field(
        default_factory=lambda: BASE_DIR / "models" / "best_plate_detector.pt"
    )
    onnx_model_path: Path = field(
        default_factory=lambda: BASE_DIR / "models" / "best_plate_detector.onnx"
    )
    prefer_onnx: bool = False
    device: str = "cpu"
    detection_conf_threshold: float = 0.25
    detection_iou_threshold: float = 0.45
    input_width: int = 1280  # Support inference size up to 1280 for small plate precision
    enable_tiled_inference: bool = False
    small_plate_mode: bool = False

    # Image Enhancement Parameters (Grayscale -> Bilateral Filter -> CLAHE -> 4x Upscale)
    upscale_factor: float = 4.0  # 4x Enlargement
    bilateral_d: int = 9
    bilateral_sigma_color: float = 75.0
    bilateral_sigma_space: float = 75.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)

    # OCR Parameters (PaddleOCR CPU Mode)
    ocr_lang: str = "en"
    use_angle_cls: bool = True
    ocr_conf_threshold: float = 0.75  # Results below 0.75 displayed as UNKNOWN

    # Zoomed Plate Panel Video Display Options
    panel_zoom_scale: float = 4.0
    panel_width: int = 340
    target_fps: float = 25.0

    def ensure_directories(self) -> None:
        """Ensure required project directories exist on disk."""
        directories = [
            self.models_dir,
            self.input_videos_dir,
            self.output_videos_dir,
            self.logs_dir,
        ]
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# Global default configuration instance
default_config = ALPRConfig()


def setup_file_logging(
    config: ALPRConfig = default_config,
    log_filename: str = "alpr_system.log",
) -> Path:
    """Configure logging to file in logs/ directory."""
    config.ensure_directories()
    log_file = config.logs_dir / log_filename

    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, logging.FileHandler) and Path(handler.baseFilename) == log_file:
            return log_file

    file_handler = logging.FileHandler(str(log_file), mode="a", encoding="utf-8")
    file_formatter = logging.Formatter(
        "%(asctime)s - [%(levelname)s] - %(name)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    file_handler.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    return log_file
