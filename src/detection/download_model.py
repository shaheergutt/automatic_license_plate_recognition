"""Pretrained Model Downloader.

Downloads pretrained YOLOv8 license plate detector weights to models/pretrained/.
"""

import logging
import urllib.request
from pathlib import Path
from typing import Optional, Union

from config import ALPRConfig, default_config

logger = logging.getLogger("ALPR.ModelDownload")

# Primary and fallback URLs for pretrained YOLOv8 license plate detector weights
DEFAULT_MODEL_URL = (
    "https://huggingface.co/keremberke/yolov8n-license-plate-detection/resolve/main/best.pt"
)
FALLBACK_MODEL_URL = (
    "https://huggingface.co/Koushim/yolov8-license-plate-detection/resolve/main/best.pt"
)


def download_pretrained_weights(
    target_path: Optional[Union[str, Path]] = None,
    config: ALPRConfig = default_config,
) -> Path:
    """Download pretrained license plate detector weights if not present.

    Args:
        target_path: Destination path for weights file.
        config: ALPRConfig instance for defaults.

    Returns:
        Path to downloaded weights file.
    """
    destination = Path(target_path or config.pretrained_model_path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists() and destination.stat().st_size > 1_000_000:
        logger.info("Pretrained model weights already exist at: %s", destination)
        return destination

    logger.info("Downloading pretrained license plate detector weights to: %s", destination)

    urls = [DEFAULT_MODEL_URL, FALLBACK_MODEL_URL]
    download_success = False

    for url in urls:
        try:
            logger.info("Attempting download from: %s", url)
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
            )
            with urllib.request.urlopen(req, timeout=60) as response, open(
                destination, "wb"
            ) as out_file:
                content = response.read()
                out_file.write(content)

            if destination.exists() and destination.stat().st_size > 1_000_000:
                logger.info(
                    "Successfully downloaded weights (Size: %.2f MB) -> %s",
                    destination.stat().st_size / (1024 * 1024),
                    destination,
                )
                download_success = True
                break
        except Exception as err:
            logger.warning("Failed download from %s: %s", url, err)

    if not download_success:
        logger.warning(
            "Could not download remote pretrained weights. Standard YOLOv8n base model will be initialized."
        )
        # Fallback to standard base ultralytics yolov8n.pt if download fails
        from ultralytics import YOLO
        base_model = YOLO("yolov8n.pt")
        base_model.save(str(destination))
        logger.info("Initialized base YOLOv8n model weights at: %s", destination)

    return destination


if __name__ == "__main__":
    download_pretrained_weights()
