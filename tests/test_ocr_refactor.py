"""Unit Tests for Accuracy-Focused OCR Refactor Modules.

Tests Image Quality Evaluation, Image Enhancement, Plate Format Validation,
Multi-Frame Majority Voting Aggregation, and Tracker Crop Memory.
"""

import os
import sys
import unittest
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.ocr.aggregator import OCRObservation, aggregate_ocr_results
from src.ocr.enhancement import enhance_crop_for_ocr
from src.ocr.image_quality import evaluate_crop_quality, is_blurry
from src.ocr.validator import clean_plate_text, validate_plate_text
from src.tracking.tracker import PlateTracker


class TestImageQuality(unittest.TestCase):
    """Test suite for image quality assessment and blur detection."""

    def test_sharp_vs_blurry_image(self):
        # Create sharp synthetic image with high edge variance
        sharp_img = np.zeros((100, 200), dtype=np.uint8)
        cv2.rectangle(sharp_img, (20, 20), (180, 80), 255, -1)
        cv2.putText(sharp_img, "LEA1234", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 0, 3)

        # Create heavily blurred version
        blurry_img = cv2.GaussianBlur(sharp_img, (25, 25), 10.0)

        quality_sharp = evaluate_crop_quality(sharp_img, blur_threshold=50.0)
        quality_blurry = evaluate_crop_quality(blurry_img, blur_threshold=50.0)

        self.assertTrue(quality_sharp.sharpness > quality_blurry.sharpness)
        self.assertTrue(quality_sharp.is_sharp)
        self.assertFalse(quality_blurry.is_sharp)

    def test_empty_crop(self):
        empty_crop = np.zeros((0, 0, 3), dtype=np.uint8)
        quality = evaluate_crop_quality(empty_crop)
        self.assertEqual(quality.overall_score, 0.0)
        self.assertFalse(quality.is_sharp)


class TestImageEnhancement(unittest.TestCase):
    """Test suite for multi-stage image enhancement pipeline."""

    def test_enhancement_output_dimensions(self):
        crop = np.zeros((30, 100, 3), dtype=np.uint8)
        cv2.putText(crop, "TEST", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        enhanced = enhance_crop_for_ocr(crop, scale=2.0, enable_denoising=False, enable_clahe=True)

        self.assertIsNotNone(enhanced)
        self.assertEqual(enhanced.shape[0], 60)
        self.assertEqual(enhanced.shape[1], 200)
        self.assertEqual(enhanced.ndim, 3)


class TestFormatValidation(unittest.TestCase):
    """Test suite for plate text cleaning and regex format validation."""

    def test_clean_text(self):
        self.assertEqual(clean_plate_text("  lea-1234! "), "LEA-1234")
        self.assertEqual(clean_plate_text("L#A@1234"), "LA1234")
        self.assertEqual(clean_plate_text("abc  5678"), "ABC-5678")

    def test_validate_plate_text(self):
        # Valid plates
        self.assertTrue(validate_plate_text("LEA1234"))
        self.assertTrue(validate_plate_text("ABC-5678"))
        self.assertTrue(validate_plate_text("ICT1122"))

        # Invalid plates
        self.assertFalse(validate_plate_text("L#A@12"))  # symbols
        self.assertFalse(validate_plate_text("A"))        # extremely short
        self.assertFalse(validate_plate_text("UNKNOWN"))  # keyword
        self.assertFalse(validate_plate_text("11111111")) # trivial repeat


class TestOCRAggregation(unittest.TestCase):
    """Test suite for multi-frame majority voting aggregation."""

    def test_majority_voting(self):
        # Simulate observations across 5 frames:
        # Frame 1 -> LEA1234 (95%)
        # Frame 2 -> LEA1234 (96%)
        # Frame 3 -> LEAI234 (78%)
        # Frame 4 -> LEA1234 (97%)
        # Frame 5 -> LEA1234 (98%)
        observations = [
            OCRObservation(text="LEA1234", confidence=0.95, crop_quality=0.8, frame_idx=1),
            OCRObservation(text="LEA1234", confidence=0.96, crop_quality=0.85, frame_idx=2),
            OCRObservation(text="LEAI234", confidence=0.78, crop_quality=0.7, frame_idx=3),
            OCRObservation(text="LEA1234", confidence=0.97, crop_quality=0.88, frame_idx=4),
            OCRObservation(text="LEA1234", confidence=0.98, crop_quality=0.9, frame_idx=5),
        ]

        winning_text, winning_conf = aggregate_ocr_results(
            observations, min_observations=2, confidence_threshold=0.75
        )

        self.assertEqual(winning_text, "LEA1234")
        self.assertGreaterEqual(winning_conf, 0.95)

    def test_low_confidence_filtering(self):
        observations = [
            OCRObservation(text="BAD001", confidence=0.50, crop_quality=0.4, frame_idx=1),
            OCRObservation(text="BAD001", confidence=0.60, crop_quality=0.45, frame_idx=2),
        ]

        winning_text, winning_conf = aggregate_ocr_results(
            observations, min_observations=2, confidence_threshold=0.75
        )

        self.assertEqual(winning_text, "UNKNOWN")


class TestTrackerMemory(unittest.TestCase):
    """Test suite for PlateTracker candidate crop buffer and observation memory."""

    def test_tracker_candidate_crops_collection(self):
        tracker = PlateTracker()
        sharp_crop = np.zeros((100, 200, 3), dtype=np.uint8)
        cv2.rectangle(sharp_crop, (20, 20), (180, 80), (255, 255, 255), -1)
        cv2.putText(sharp_crop, "LEA1234", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)

        accepted, quality, sharpness = tracker.add_crop_and_evaluate(
            track_id=1, crop=sharp_crop, frame_idx=1
        )

        self.assertTrue(accepted)
        state = tracker.tracks[1]
        self.assertEqual(len(state.candidate_crops), 1)

        # Record observation
        tracker.record_ocr_observation(
            track_id=1,
            text="LEA1234",
            confidence=0.984,
            crop_quality=quality,
            sharpness=sharpness,
            frame_idx=1,
            crop=sharp_crop,
        )

        self.assertEqual(state.best_text, "LEA1234")
        self.assertAlmostEqual(state.best_ocr_confidence, 0.984, places=3)


if __name__ == "__main__":
    unittest.main()
