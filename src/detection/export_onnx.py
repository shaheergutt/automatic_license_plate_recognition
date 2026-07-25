"""Utility script and helper to export PyTorch YOLO models to ONNX format for CPU inference."""

import logging
from pathlib import Path
from typing import Optional, Union

from ultralytics import YOLO

logger = logging.getLogger("ALPR.ExportONNX")


def export_yolo_to_onnx(
    model_path: Union[str, Path],
    output_onnx_path: Optional[Union[str, Path]] = None,
    imgsz: int = 640,
    dynamic: bool = True,
) -> Path:
    """Export a YOLO PyTorch (.pt) model file to ONNX (.onnx) format.

    Args:
        model_path: Path to PyTorch model weights (.pt).
        output_onnx_path: Destination path for exported ONNX file.
        imgsz: Image size for ONNX export.
        dynamic: Enable dynamic axes for variable input batch/resolution.

    Returns:
        Path to exported ONNX model file.
    """
    model_p = Path(model_path)
    if not model_p.exists():
        raise FileNotFoundError(f"PyTorch model file not found: {model_p}")

    logger.info("Exporting PyTorch model '%s' to ONNX format...", model_p)

    model = YOLO(str(model_p))
    exported_file = model.export(
        format="onnx",
        imgsz=imgsz,
        dynamic=dynamic,
        simplify=True,
        opset=12,
    )

    exported_path = Path(exported_file)
    logger.info("Successfully exported ONNX model to: %s", exported_path)

    if output_onnx_path:
        target_path = Path(output_onnx_path)
        if exported_path != target_path:
            target_path.parent.mkdir(parents=True, exist_ok=True)
            if target_path.exists():
                target_path.unlink()
            exported_path.rename(target_path)
            exported_path = target_path

    return exported_path


if __name__ == "__main__":
    import sys
    from config import default_config

    pt_model = default_config.model_path
    onnx_model = default_config.onnx_model_path

    if pt_model.exists():
        out = export_yolo_to_onnx(pt_model, onnx_model)
        print(f"Export completed: {out}")
    else:
        print(f"Model file not found: {pt_model}")
        sys.exit(1)
