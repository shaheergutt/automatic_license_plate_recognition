"""OCR Engine Module using PaddleOCR on CPU.

Provides license plate text recognition, 4x image enhancement, format validation,
and confidence thresholding. If confidence is below threshold, returns 'UNKNOWN'.
"""

import logging
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

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

import numpy as np
from paddleocr import PaddleOCR

from config import ALPRConfig, default_config
from src.ocr.enhancement import enhance_crop_for_ocr
from src.ocr.validator import clean_plate_text, validate_plate_text

logger = logging.getLogger("ALPR.OCR")


@dataclass
class OCRResult:
    """Dataclass holding OCR recognition result for a cropped license plate."""

    text: str
    confidence: float
    raw_text: str
    is_valid: bool


class OCREngine:
    """Optical Character Recognition engine using PaddleOCR in CPU mode."""

    def __init__(
        self,
        lang: Optional[str] = None,
        use_angle_cls: Optional[bool] = None,
        ocr_conf_threshold: Optional[float] = None,
        config: ALPRConfig = default_config,
    ) -> None:
        """Initialize PaddleOCR engine.

        Args:
            lang: Language string ('en').
            use_angle_cls: Enable orientation angle classification.
            ocr_conf_threshold: Min confidence threshold for valid output (0.75).
            config: ALPRConfig instance.
        """
        self.config = config
        self.lang = lang or config.ocr_lang
        self.use_angle_cls = (
            use_angle_cls if use_angle_cls is not None else config.use_angle_cls
        )
        self.ocr_conf_threshold = (
            ocr_conf_threshold
            if ocr_conf_threshold is not None
            else config.ocr_conf_threshold
        )

        logger.info(
            "Initializing PaddleOCR (lang='%s', use_angle_cls=%s, conf_thresh=%.2f)",
            self.lang,
            self.use_angle_cls,
            self.ocr_conf_threshold,
        )

        try:
            self.ocr_engine = PaddleOCR(
                use_angle_cls=self.use_angle_cls,
                lang=self.lang,
                enable_mkldnn=False,
            )
        except Exception:
            self.ocr_engine = PaddleOCR(lang=self.lang)

        logger.info("PaddleOCR engine initialized successfully on CPU.")

    def recognize(self, crop: np.ndarray) -> OCRResult:
        """Perform 4x image enhancement and PaddleOCR text extraction.

        Args:
            crop: BGR image crop of license plate.

        Returns:
            OCRResult object with text (or 'UNKNOWN'), confidence, and validity status.
        """
        if crop is None or crop.size == 0 or crop.shape[0] < 6 or crop.shape[1] < 12:
            return OCRResult(text="UNKNOWN", confidence=0.0, raw_text="", is_valid=False)

        # 1. Image Enhancement Preprocessing (Grayscale -> Bilateral Filter -> CLAHE -> 4x Upscale)
        enhanced_crop = enhance_crop_for_ocr(
            crop, scale=getattr(self.config, "upscale_factor", 4.0)
        )

        # 2. PaddleOCR Recognition
        try:
            results = self.ocr_engine.ocr(enhanced_crop)
        except Exception as err:
            logger.warning("PaddleOCR recognition error: %s", err)
            return OCRResult(text="UNKNOWN", confidence=0.0, raw_text="", is_valid=False)

        if not results:
            return OCRResult(text="UNKNOWN", confidence=0.0, raw_text="", is_valid=False)

        texts = []
        confidences = []

        if isinstance(results, list) and len(results) > 0 and isinstance(results[0], dict):
            res_dict = results[0]
            rec_texts = res_dict.get("rec_texts", [])
            rec_scores = res_dict.get("rec_scores", [])
            for txt, conf in zip(rec_texts, rec_scores):
                cleaned = clean_plate_text(str(txt))
                if cleaned:
                    texts.append(cleaned)
                    confidences.append(float(conf))
        elif isinstance(results, list) and len(results) > 0 and isinstance(results[0], list):
            for line in results[0]:
                if len(line) >= 2 and isinstance(line[1], (tuple, list)):
                    txt, conf = line[1][0], float(line[1][1])
                    cleaned = clean_plate_text(str(txt))
                    if cleaned:
                        texts.append(cleaned)
                        confidences.append(conf)

        if not texts:
            return OCRResult(text="UNKNOWN", confidence=0.0, raw_text="", is_valid=False)

        full_raw_text = "-".join(texts)
        avg_confidence = float(np.mean(confidences))

        # Check confidence thresholding and plate regex format validation
        is_valid = (
            avg_confidence >= self.ocr_conf_threshold
        ) and validate_plate_text(full_raw_text)

        if not is_valid:
            logger.debug(
                "OCR text '%s' confidence (%.2f) below threshold (%.2f) or invalid format.",
                full_raw_text,
                avg_confidence,
                self.ocr_conf_threshold,
            )
            return OCRResult(
                text="UNKNOWN",
                confidence=avg_confidence,
                raw_text=full_raw_text,
                is_valid=False,
            )

        return OCRResult(
            text=full_raw_text,
            confidence=avg_confidence,
            raw_text=full_raw_text,
            is_valid=True,
        )
