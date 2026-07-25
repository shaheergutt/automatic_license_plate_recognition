"""Unit Tests for Simple Accurate ALPR System.

Tests 4x Image Enhancement, Format Validation, OCR Confidence Thresholding,
Zoomed Plate Panel Video Drawing, and Results Export.
"""

import os
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ocr.enhancement import enhance_crop_for_ocr
from src.ocr.validator import clean_plate_text, validate_plate_text
from src.utils.panel_drawer import (
    DetectedPlateItem,
    draw_annotated_frame_with_zoomed_panels,
)
from src.pipeline.video_pipeline import SimpleALPRPipeline


class Test4xEnhancement(unittest.TestCase):
    """Test suite for 4x image enhancement pipeline."""

    def test_enhancement_pipeline(self):
        crop = np.zeros((30, 100, 3), dtype=np.uint8)
        cv2.putText(crop, "LEA1234", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        enhanced = enhance_crop_for_ocr(crop, scale=4.0)

        self.assertIsNotNone(enhanced)
        self.assertEqual(enhanced.shape[0], 120)  # 30 * 4 = 120
        self.assertEqual(enhanced.shape[1], 400)  # 100 * 4 = 400
        self.assertEqual(enhanced.ndim, 3)


class TestFormatValidation(unittest.TestCase):
    """Test suite for plate text cleaning and format validation."""

    def test_clean_text(self):
        self.assertEqual(clean_plate_text("  lea-1234! "), "LEA-1234")
        self.assertEqual(clean_plate_text("L#A@1234"), "LA1234")

    def test_validate_text(self):
        self.assertTrue(validate_plate_text("LEA-1234"))
        self.assertTrue(validate_plate_text("ABC-5678"))
        self.assertTrue(validate_plate_text("ICT-1122"))

        self.assertFalse(validate_plate_text("UNKNOWN"))
        self.assertFalse(validate_plate_text("A"))


class TestZoomedPanelDrawer(unittest.TestCase):
    """Test suite for Zoomed Plate Panel video frame drawing."""

    def test_panel_rendering(self):
        frame = np.full((480, 640, 3), 100, dtype=np.uint8)
        crop = np.full((30, 90, 3), 200, dtype=np.uint8)

        detections = [
            DetectedPlateItem(
                box=(50, 50, 140, 80),
                crop=crop,
                text="LEA-1234",
                confidence=0.987,
                plate_idx=1,
            )
        ]

        canvas = draw_annotated_frame_with_zoomed_panels(frame, detections, sidebar_width=360)

        self.assertIsNotNone(canvas)
        self.assertEqual(canvas.shape[0], 480)
        self.assertEqual(canvas.shape[1], 640 + 360)  # Width extended with sidebar


class TestResultsReportExport(unittest.TestCase):
    """Test suite for results.txt summary report export."""

    def test_report_generation(self):
        tmp_report = Path(__file__).parent / "tmp_results.txt"
        unique_plates = {
            "LEA-1234": 0.987,
            "ABC-5678": 0.964,
            "ICT-1122": 0.991,
        }

        try:
            SimpleALPRPipeline.export_results_report(tmp_report, unique_plates)

            content = tmp_report.read_text(encoding="utf-8")
            self.assertIn("DETECTED LICENSE PLATES", content)
            self.assertIn("LEA-1234    98.7%", content)
            self.assertIn("ABC-5678    96.4%", content)
            self.assertIn("ICT-1122    99.1%", content)
        finally:
            if tmp_report.exists():
                tmp_report.unlink()


if __name__ == "__main__":
    unittest.main()
