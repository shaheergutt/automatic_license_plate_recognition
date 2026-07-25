"""Zoomed Plate Panel Video Drawing and Visualization Module.

Renders high-visibility bounding boxes on the main video frame and constructs a professional
side-by-side or overlay Zoomed Plate Panel showing 4x-6x enlarged plate crops, extracted text,
and confidence scores for demonstration videos.
"""

from typing import List, Tuple
import cv2
import numpy as np


class DetectedPlateItem:
    """Dataclass holding detection box, crop image, text, and confidence for panel rendering."""

    def __init__(
        self,
        box: Tuple[float, float, float, float],
        crop: np.ndarray,
        text: str,
        confidence: float,
        plate_idx: int = 1,
    ):
        self.x1, self.y1, self.x2, self.y2 = box
        self.crop = crop
        self.text = text
        self.confidence = confidence
        self.plate_idx = plate_idx


def draw_annotated_frame_with_zoomed_panels(
    frame: np.ndarray,
    detections: List[DetectedPlateItem],
    sidebar_width: int = 360,
) -> np.ndarray:
    """Render bounding boxes on video frame and attach a professional Zoomed Plate Panel sidebar.

    Args:
        frame: BGR video frame image array.
        detections: List of DetectedPlateItem objects.
        sidebar_width: Width of right-side panel in pixels.

    Returns:
        Combined annotated BGR video frame array (Width = frame_width + sidebar_width).
    """
    if frame is None or frame.size == 0:
        return frame

    h, w = frame.shape[:2]
    annotated_frame = frame.copy()

    # 1. Draw Bounding Boxes on Original Video Frame
    for idx, item in enumerate(detections, 1):
        x1, y1, x2, y2 = int(item.x1), int(item.y1), int(item.x2), int(item.y2)

        if item.text != "UNKNOWN":
            box_color = (0, 220, 0)      # Bright Green
            bg_color = (0, 160, 0)
        else:
            box_color = (0, 200, 255)    # Yellow / Gold
            bg_color = (0, 140, 200)

        # Bounding box
        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), box_color, 3)

        # Label tag on box
        conf_pct = item.confidence * 100.0
        if item.text != "UNKNOWN":
            label = f"#{idx}: {item.text} ({conf_pct:.1f}%)"
        else:
            label = f"#{idx}: UNKNOWN"

        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        ly1 = max(0, y1 - th - 12)
        cv2.rectangle(annotated_frame, (x1, ly1), (x1 + tw + 10, y1), bg_color, -1)
        cv2.putText(
            annotated_frame,
            label,
            (x1 + 5, y1 - 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    # 2. Construct Professional Zoomed Plate Panel Sidebar
    sidebar = np.full((h, sidebar_width, 3), (24, 24, 28), dtype=np.uint8)

    # Panel Header Banner
    cv2.rectangle(sidebar, (0, 0), (sidebar_width, 50), (35, 35, 42), -1)
    cv2.putText(
        sidebar,
        "ZOOMED PLATE PANELS",
        (15, 33),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.line(sidebar, (0, 50), (sidebar_width, 50), (60, 60, 70), 1)

    current_y = 65

    if not detections:
        cv2.putText(
            sidebar,
            "No plates detected",
            (20, current_y + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (140, 140, 140),
            1,
            cv2.LINE_AA,
        )
    else:
        for idx, item in enumerate(detections, 1):
            if current_y + 180 > h:
                break  # Prevent overflowing frame height

            card_h = 175
            card_w = sidebar_width - 30
            card_x = 15

            # Card background
            cv2.rectangle(
                sidebar,
                (card_x, current_y),
                (card_x + card_w, current_y + card_h),
                (38, 38, 45),
                -1,
            )
            cv2.rectangle(
                sidebar,
                (card_x, current_y),
                (card_x + card_w, current_y + card_h),
                (70, 70, 80),
                1,
            )

            # Card Header
            cv2.putText(
                sidebar,
                f"PLATE #{idx}",
                (card_x + 12, current_y + 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (200, 200, 200),
                1,
                cv2.LINE_AA,
            )

            # Enlarged Plate Crop (4x-6x preview scaling to fit card width)
            crop = item.crop
            if crop is not None and crop.size > 0:
                crop_h, crop_w = crop.shape[:2]
                target_crop_w = card_w - 24
                scale = target_crop_w / crop_w
                target_crop_h = int(crop_h * scale)
                target_crop_h = min(target_crop_h, 75)

                zoomed_crop = cv2.resize(
                    crop, (target_crop_w, target_crop_h), interpolation=cv2.INTER_CUBIC
                )

                # Insert zoomed crop image into sidebar card
                crop_y1 = current_y + 32
                crop_y2 = crop_y1 + target_crop_h
                crop_x1 = card_x + 12
                crop_x2 = crop_x1 + target_crop_w

                sidebar[crop_y1:crop_y2, crop_x1:crop_x2] = zoomed_crop
                cv2.rectangle(
                    sidebar,
                    (crop_x1, crop_y1),
                    (crop_x2, crop_y2),
                    (255, 255, 255),
                    1,
                )

                text_y = crop_y2 + 25
            else:
                text_y = current_y + 80

            # Render Extracted Text and Confidence Score
            if item.text != "UNKNOWN":
                text_color = (0, 230, 0)
                text_str = item.text
            else:
                text_color = (0, 200, 255)
                text_str = "UNKNOWN"

            conf_pct = item.confidence * 100.0
            conf_str = f"{conf_pct:.1f}%" if item.text != "UNKNOWN" else "Low Conf"

            cv2.putText(
                sidebar,
                text_str,
                (card_x + 12, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                text_color,
                2,
                cv2.LINE_AA,
            )

            cv2.putText(
                sidebar,
                conf_str,
                (card_x + card_w - 90, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            current_y += card_h + 15

    # 3. Combine Main Annotated Frame and Zoomed Panel Sidebar Side-by-Side
    canvas = np.hstack((annotated_frame, sidebar))
    return canvas
