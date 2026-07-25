"""Focused CLI Entry Point for Simple Accurate Automatic License Plate Recognition (ALPR).

Executes plate detection, memory cropping, 4x image enhancement, PaddleOCR text extraction,
Zoomed Plate Panel video rendering, saving annotated output video, and exporting results.txt.
"""

import argparse
import importlib
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Disable OneDNN / MKLDNN PIR executor flags for Paddle 3.x stability on CPU
os.environ["FLAGS_use_onednn"] = "0"
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["FLAGS_enable_pir_in_executor"] = "0"
os.environ["FLAGS_use_mkldnn"] = "0"

# Support custom target path for PaddleOCR on Windows if installed there
custom_paddle_dir = Path.home() / "paddle_lib"
pylib_dir = Path("C:/pylib/Python313/site-packages")
for p_dir in [custom_paddle_dir, pylib_dir]:
    if p_dir.exists() and str(p_dir) not in sys.path:
        sys.path.insert(0, str(p_dir))

from config import ALPRConfig, default_config, setup_file_logging
from src.pipeline.video_pipeline import SimpleALPRPipeline

logger = logging.getLogger("ALPR.Main")


def check_dependencies(packages: List[str]) -> Dict[str, bool]:
    """Check whether required Python packages are available."""
    results: Dict[str, bool] = {}
    for package in packages:
        try:
            importlib.import_module(package)
            results[package] = True
            logger.info("  [✓] %s: Available", package)
        except ImportError as err:
            results[package] = False
            logger.warning("  [!] %s: Not found (%s)", package, err)
    return results


def select_input_video(
    videos_dir: Path,
    cli_video: Optional[str] = None,
) -> Path:
    """Select target input video file via CLI argument or automatic discovery."""
    if cli_video:
        video_p = Path(cli_video)
        if video_p.exists():
            return video_p
        logger.warning("Specified CLI video path not found: %s", cli_video)

    videos_dir.mkdir(parents=True, exist_ok=True)
    video_extensions = {".mp4", ".avi", ".mov", ".mkv"}
    available_videos = sorted(
        [f for f in videos_dir.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]
    )

    if not available_videos:
        logger.error("No input videos found in: %s", videos_dir)
        print(f"\nERROR: No input videos (.mp4, .avi, .mov, .mkv) found in {videos_dir}")
        print("Please place video files into videos/input/ and retry.")
        sys.exit(1)

    return available_videos[0]


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Focused Simple Accurate Automatic License Plate Recognition (ALPR)"
    )
    parser.add_argument(
        "--video",
        type=str,
        default=None,
        help="Path to input video file (.mp4, .avi, .mov, .mkv)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Path to save annotated output video file (default: videos/output/annotated_output.mp4)",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=None,
        help="Path to save results text report (default: results.txt)",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=None,
        help="Detection confidence threshold (default: 0.25)",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Perform dependency and setup checks only without processing video",
    )

    args = parser.parse_args()

    print("=" * 65)
    print(" Simple Accurate Automatic License Plate Recognition (ALPR)")
    print("=" * 65)

    # 1. Initialize Logging
    try:
        setup_file_logging(config=default_config)
    except Exception as err:
        print(f"Warning: Logging setup error: {err}")

    config: ALPRConfig = default_config
    if args.conf is not None:
        config.detection_conf_threshold = args.conf
    config.ensure_directories()

    # 2. Check Core Dependencies
    logger.info("Checking system dependencies...")
    required_modules = [
        "ultralytics",
        "cv2",
        "paddleocr",
        "numpy",
    ]
    dep_results = check_dependencies(required_modules)
    all_ok = all(dep_results.values())

    if not all_ok:
        missing = [pkg for pkg, ok in dep_results.items() if not ok]
        print(f"\nERROR: Missing required dependencies: {', '.join(missing)}")
        return 1

    if args.check_only:
        print("\n[✓] Environment and dependency checks completed successfully.")
        return 0

    # 3. Select Input Video
    input_video = select_input_video(
        videos_dir=config.input_videos_dir,
        cli_video=args.video,
    )

    output_video = (
        Path(args.output)
        if args.output
        else config.output_video_path
    )
    results_txt = (
        Path(args.results)
        if args.results
        else config.results_txt_path
    )

    print(f"\nInput Video  : {input_video}")
    print(f"Output Video : {output_video}")
    print(f"Results File : {results_txt}\n")

    # 4. Initialize and Execute Simple ALPR Pipeline
    try:
        pipeline = SimpleALPRPipeline(config=config)
        pipeline.process_video(
            input_path=input_video,
            output_path=output_video,
            results_txt_path=results_txt,
        )
        return 0
    except Exception as err:
        logger.critical("ALPR processing failed: %s", err, exc_info=True)
        print(f"\nERROR: ALPR processing failed: {err}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
