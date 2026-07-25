"""Focused Sequential Video Pipeline Module for Simple Accurate ALPR System.

Performs frame-by-frame license plate detection, memory crop extraction, 4x image enhancement,
PaddleOCR text extraction, zoomed plate panel video rendering, saving annotated output video,
and exporting results.txt final summary.
"""

import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from config import ALPRConfig, default_config
from src.detection.plate_detector import LicensePlateDetector
from src.ocr.ocr_engine import OCREngine
from src.utils.panel_drawer import (
    DetectedPlateItem,
    draw_annotated_frame_with_zoomed_panels,
)

logger = logging.getLogger("ALPR.Pipeline")


class SimpleALPRPipeline:
    """Sequential ALPR Video Pipeline."""

    def __init__(
        self,
        detector: Optional[LicensePlateDetector] = None,
        ocr_engine: Optional[OCREngine] = None,
        config: ALPRConfig = default_config,
    ) -> None:
        """Initialize ALPR detector and OCR engine.

        Args:
            detector: LicensePlateDetector instance.
            ocr_engine: OCREngine instance.
            config: ALPRConfig instance.
        """
        self.config = config
        self.detector = detector or LicensePlateDetector(config=config)
        self.ocr_engine = ocr_engine or OCREngine(config=config)

    def process_video(
        self,
        input_path: Union[str, Path],
        output_path: Optional[Union[str, Path]] = None,
        results_txt_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, float]:
        """Process input video frame-by-frame and save annotated output video & results.txt.

        Args:
            input_path: Path to input video file.
            output_path: Destination path for annotated MP4 video file.
            results_txt_path: Destination path for results.txt report.

        Returns:
            Dictionary mapping unique plate text to highest confidence score.
        """
        input_p = Path(input_path)
        if not input_p.exists():
            raise FileNotFoundError(f"Input video file not found: {input_p}")

        output_p = (
            Path(output_path)
            if output_path
            else self.config.output_video_path
        )
        results_p = (
            Path(results_txt_path)
            if results_txt_path
            else self.config.results_txt_path
        )

        output_p.parent.mkdir(parents=True, exist_ok=True)
        results_p.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Opening input video: %s", input_p.name)
        cap = cv2.VideoCapture(str(input_p))

        if not cap.isOpened():
            raise RuntimeError(f"Could not open input video: {input_p}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        orig_fps = cap.get(cv2.CAP_PROP_FPS) or self.config.target_fps

        writer: Optional[cv2.VideoWriter] = None
        unique_plates: Dict[str, float] = {}

        frame_idx = 0
        t_start = time.perf_counter()

        logger.info("Processing video frames and rendering Zoomed Plate Panels...")

        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            h, w = frame.shape[:2]

            # 1. License Plate Detection (Inference size up to 1280px)
            detections = self.detector.detect(frame)

            plate_items: List[DetectedPlateItem] = []

            # 2. Crop & OCR Step per detected plate
            for idx, det in enumerate(detections, 1):
                x1, y1 = max(0, int(det.x1)), max(0, int(det.y1))
                x2, y2 = min(w, int(det.x2)), min(h, int(det.y2))

                if (x2 - x1) > 8 and (y2 - y1) > 8:
                    # Crop plate region into memory
                    crop = frame[y1:y2, x1:x2].copy()

                    # Perform 4x enhancement and PaddleOCR recognition
                    ocr_res = self.ocr_engine.recognize(crop)

                    if ocr_res.is_valid and ocr_res.text != "UNKNOWN":
                        # Record unique plate and keep highest confidence score
                        if (
                            ocr_res.text not in unique_plates
                            or ocr_res.confidence > unique_plates[ocr_res.text]
                        ):
                            unique_plates[ocr_res.text] = ocr_res.confidence

                    plate_items.append(
                        DetectedPlateItem(
                            box=(det.x1, det.y1, det.x2, det.y2),
                            crop=crop,
                            text=ocr_res.text,
                            confidence=ocr_res.confidence,
                            plate_idx=idx,
                        )
                    )

            # 3. Render Annotated Frame with Zoomed Plate Panels
            annotated_canvas = draw_annotated_frame_with_zoomed_panels(
                frame=frame,
                detections=plate_items,
                sidebar_width=self.config.panel_width,
            )

            # 4. Write Frame to Output Video
            if writer is None:
                canvas_h, canvas_w = annotated_canvas.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(
                    str(output_p), fourcc, orig_fps, (canvas_w, canvas_h)
                )

            writer.write(annotated_canvas)

            if frame_idx % 30 == 0 or frame_idx == total_frames:
                logger.info("Processed frame %d/%d...", frame_idx, total_frames)

        cap.release()
        if writer is not None:
            writer.release()

        t_elapsed = time.perf_counter() - t_start
        logger.info(
            "Video processing complete (%d frames in %.1fs). Output: %s",
            frame_idx,
            t_elapsed,
            output_p,
        )

        # 5. Export Final Summary Report (results.txt)
        self.export_results_report(results_p, unique_plates)
        self.print_summary_report(unique_plates)

        return unique_plates

    @staticmethod
    def export_results_report(
        results_file: Path, unique_plates: Dict[str, float]
    ) -> None:
        """Generate and save results.txt formatted to exact prompt specification.

        Args:
            results_file: Output path for results.txt.
            unique_plates: Dict mapping unique plate text to highest confidence score.
        """
        lines = []
        lines.append("==================================")
        lines.append("DETECTED LICENSE PLATES")
        lines.append("==================================\n")

        if not unique_plates:
            lines.append("No license plates detected.\n")
        else:
            for text, conf in unique_plates.items():
                conf_pct = conf * 100.0
                lines.append(f"{text:<10} {conf_pct:5.1f}%\n")

        lines.append("==================================\n")

        results_file.parent.mkdir(parents=True, exist_ok=True)
        results_file.write_text("\n".join(lines), encoding="utf-8")
        logger.info("Saved final results report to: %s", results_file)

    @staticmethod
    def print_summary_report(unique_plates: Dict[str, float]) -> None:
        """Print final summary report to stdout."""
        print("\n" + "=" * 34)
        print("DETECTED LICENSE PLATES")
        print("=" * 34 + "\n")

        if not unique_plates:
            print("No license plates detected.")
        else:
            for text, conf in unique_plates.items():
                conf_pct = conf * 100.0
                print(f"{text:<10} {conf_pct:5.1f}%\n")

        print("=" * 34 + "\n")
