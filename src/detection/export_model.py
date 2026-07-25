"""Model Export Utility.

Copies trained best checkpoint from experiments/experiment_001/weights/best.pt
to models/best_plate_detector.pt for drop-in pipeline replacement.
"""

import logging
import shutil
from pathlib import Path

from config import ALPRConfig, default_config

logger = logging.getLogger("ALPR.Export")


def export_best_model(config: ALPRConfig = default_config) -> Path:
    """Copy best experiment weights to models/best_plate_detector.pt.

    Args:
        config: ALPRConfig instance.

    Returns:
        Path to exported model weights.
    """
    src_weights = config.experiment_dir / "weights" / "best.pt"
    if not src_weights.exists():
        raise FileNotFoundError(f"Trained weights not found at: {src_weights}")

    dst_weights = config.custom_model_path
    dst_weights.parent.mkdir(parents=True, exist_ok=True)

    shutil.copy(src_weights, dst_weights)
    logger.info("Successfully exported custom model -> %s (Size: %.2f MB)", dst_weights, dst_weights.stat().st_size / (1024 * 1024))
    return dst_weights


if __name__ == "__main__":
    export_best_model()
