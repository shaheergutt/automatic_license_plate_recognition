"""License Plate Format Validation and Text Sanitization Module.

Provides regex-based text cleaning, artifact removal, and validation rules
to ensure only plausible license plate numbers pass as final output.
"""

import logging
import re
from typing import Optional

logger = logging.getLogger("ALPR.Validator")


def clean_plate_text(raw_text: str) -> str:
    """Sanitize raw OCR output string.

    Converts to uppercase, strips non-alphanumeric characters (except single hyphens/spaces),
    and standardizes spacing into hyphens.

    Args:
        raw_text: Raw string output from OCR engine.

    Returns:
        Cleaned uppercase license plate text string.
    """
    if not raw_text:
        return ""

    # Remove special symbols, keeping alphanumeric, hyphens, spaces
    cleaned = re.sub(r"[^A-Za-z0-9\- ]", "", raw_text)
    # Standardize whitespace or multiple hyphens into single hyphen
    cleaned = re.sub(r"[\s\-]+", "-", cleaned.strip()).upper()
    # Remove leading or trailing hyphens
    cleaned = cleaned.strip("-")

    return cleaned


def validate_plate_text(
    text: str,
    min_length: int = 4,
    max_length: int = 12,
) -> bool:
    """Validate whether sanitized plate text meets license plate structural constraints.

    Validation Rules:
    - Must not be empty or "UNKNOWN".
    - Length must be between min_length and max_length (e.g., 4 to 12 chars).
    - Must contain at least one digit or letter (not just hyphens).
    - Must not contain invalid repeated character patterns (e.g., 'AAAAAA', '000000').
    - Must pass regex check for valid alphanumeric plate structure.

    Args:
        text: Sanitized license plate text.
        min_length: Minimum valid text length (default 4).
        max_length: Maximum valid text length (default 12).

    Returns:
        Boolean indicating whether string is a valid plate format.
    """
    if not text or text.upper() == "UNKNOWN":
        return False

    # Strip hyphens for length checking
    alphanumeric_only = re.sub(r"[^A-Z0-9]", "", text)

    # 1. Length constraint
    if len(alphanumeric_only) < min_length or len(alphanumeric_only) > max_length:
        logger.debug("Validation failed: Text '%s' length %d outside bounds [%d, %d]", text, len(alphanumeric_only), min_length, max_length)
        return False

    # 2. Must satisfy valid alphanumeric plate structure (letters and digits, optional hyphens)
    pattern = r"^[A-Z0-9]{1,12}(?:-[A-Z0-9]{1,12}){0,2}$"
    if not re.match(pattern, text):
        logger.debug("Validation failed: Text '%s' does not match regex pattern", text)
        return False

    # 3. Reject repetitive single-character sequences (e.g., "IIII", "1111", "XXXX")
    if len(set(alphanumeric_only)) == 1 and len(alphanumeric_only) >= 4:
        logger.debug("Validation failed: Text '%s' contains trivial repeated characters", text)
        return False

    return True
