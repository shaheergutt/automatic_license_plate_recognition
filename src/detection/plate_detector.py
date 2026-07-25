"""License Plate Detection Module with ONNX CPU Acceleration & PyTorch Fallback.

Supports YOLOv8 license plate detector with automatic ONNX Runtime loading on CPU.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import cv2
import numpy as np

from config import ALPRConfig, default_config
from src.detection.export_onnx import export_yolo_to_onnx

logger = logging.getLogger("ALPR.Detection")


@dataclass
class DetectionBox:
    """Dataclass representing a single detected license plate bounding box."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int = 0
    class_name: str = "License Plate"

    @property
    def box_xyxy(self) -> Tuple[float, float, float, float]:
        """Return coordinates as (x1, y1, x2, y2) tuple."""
        return (self.x1, self.y1, self.x2, self.y2)

    @property
    def area(self) -> float:
        """Calculate bounding box area in pixels."""
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


class LicensePlateDetector:
    """Detector for locating license plates using PyTorch or ONNX Runtime on CPU."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        onnx_model_path: Optional[Union[str, Path]] = None,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        input_width: Optional[int] = None,
        config: ALPRConfig = default_config,
    ) -> None:
        """Initialize detector with ONNX Runtime acceleration and PyTorch fallback.

        Args:
            model_path: Path to PyTorch .pt model.
            onnx_model_path: Path to ONNX .onnx model.
            conf_threshold: Detection confidence threshold.
            iou_threshold: NMS IOU threshold.
            input_width: Detection input resize width.
            config: ALPRConfig instance.
        """
        self.config = config
        self.model_path = Path(model_path or config.model_path)
        self.onnx_model_path = Path(onnx_model_path or config.onnx_model_path)
        self.conf_threshold = (
            conf_threshold if conf_threshold is not None else config.detection_conf_threshold
        )
        self.iou_threshold = (
            iou_threshold if iou_threshold is not None else config.detection_iou_threshold
        )
        self.input_width = input_width or config.input_width

        self.use_onnx = False
        self.onnx_session = None
        self.pytorch_model = None

        # Attempt to load ONNX Runtime if requested and available
        if config.prefer_onnx:
            self._init_onnx_engine()

        # Fallback to PyTorch if ONNX is not available or failed to load
        if not self.use_onnx:
            self._init_pytorch_engine()

    def _init_onnx_engine(self) -> None:
        """Attempt to initialize ONNX Runtime inference session."""
        try:
            import onnxruntime as ort

            # Auto-export PyTorch model to ONNX if .onnx does not exist
            if not self.onnx_model_path.exists() and self.model_path.exists():
                logger.info("ONNX model missing. Exporting PyTorch model to ONNX...")
                try:
                    export_yolo_to_onnx(
                        model_path=self.model_path,
                        output_onnx_path=self.onnx_model_path,
                        imgsz=self.input_width,
                    )
                except Exception as export_err:
                    logger.warning("Failed to auto-export ONNX model: %s", export_err)

            if self.onnx_model_path.exists():
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 4
                opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                self.onnx_session = ort.InferenceSession(
                    str(self.onnx_model_path),
                    providers=["CPUExecutionProvider"],
                    sess_options=opts,
                )
                self.use_onnx = True
                logger.info("Loaded ONNX Runtime engine: %s", self.onnx_model_path)
            else:
                logger.info("ONNX model file not found at %s.", self.onnx_model_path)
        except ImportError:
            logger.info("onnxruntime module not installed. Defaulting to PyTorch.")
        except Exception as err:
            logger.warning("Failed to initialize ONNX Runtime session: %s", err)

    def _init_pytorch_engine(self) -> None:
        """Initialize PyTorch YOLO inference engine."""
        from ultralytics import YOLO

        if not self.model_path.exists():
            logger.error("PyTorch model weights file not found: %s", self.model_path)
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")

        logger.info("Loading PyTorch YOLO model: %s", self.model_path)
        self.pytorch_model = YOLO(str(self.model_path))
        self.use_onnx = False
        logger.info("PyTorch YOLO detector loaded successfully.")

    def detect(self, frame: np.ndarray) -> List[DetectionBox]:
        """Detect license plates in a single frame.

        Args:
            frame: Input image/frame in BGR format.

        Returns:
            List of DetectionBox objects.
        """
        if frame is None or frame.size == 0:
            return []

        if self.config.enable_tiled_inference or self.config.small_plate_mode:
            return self.detect_tiled(
                frame,
                tile_size=self.config.tile_size,
                tile_overlap=self.config.tile_overlap,
            )

        return self._detect_raw(frame)

    def _detect_raw(self, frame: np.ndarray) -> List[DetectionBox]:
        """Execute single pass inference using ONNX or PyTorch."""
        if self.use_onnx and self.onnx_session is not None:
            return self._detect_onnx(frame)
        elif self.pytorch_model is not None:
            return self._detect_pytorch(frame)
        else:
            logger.error("No inference engine available.")
            return []

    def detect_tiled(
        self,
        frame: np.ndarray,
        tile_size: int = 640,
        tile_overlap: float = 0.20,
    ) -> List[DetectionBox]:
        """Run tiled inference over frame to detect small/distant license plates.

        Workflow:
          Frame -> Split into overlapping tiles -> Run detector on each tile -> Merge with NMS

        Args:
            frame: Input image/frame in BGR format.
            tile_size: Tile width and height in pixels.
            tile_overlap: Overlap fraction between neighboring tiles.

        Returns:
            List of merged DetectionBox objects.
        """
        if frame is None or frame.size == 0:
            return []

        orig_h, orig_w = frame.shape[:2]

        # 1. Full Frame Detection
        full_dets = self._detect_raw(frame)

        all_boxes = []
        all_scores = []

        for det in full_dets:
            all_boxes.append([det.x1, det.y1, det.x2 - det.x1, det.y2 - det.y1])
            all_scores.append(det.confidence)

        # 2. Overlapping Grid Tiles
        stride = int(tile_size * (1.0 - tile_overlap))
        if stride <= 0:
            stride = tile_size

        y_max = max(1, orig_h - tile_size // 2)
        x_max = max(1, orig_w - tile_size // 2)

        for y in range(0, y_max, stride):
            for x in range(0, x_max, stride):
                x2 = min(orig_w, x + tile_size)
                y2 = min(orig_h, y + tile_size)
                x1 = max(0, x2 - tile_size)
                y1 = max(0, y2 - tile_size)

                tile = frame[y1:y2, x1:x2]
                if tile.shape[0] < 40 or tile.shape[1] < 40:
                    continue

                tile_dets = self._detect_raw(tile)
                for det in tile_dets:
                    abs_x1 = float(x1 + det.x1)
                    abs_y1 = float(y1 + det.y1)
                    abs_x2 = float(x1 + det.x2)
                    abs_y2 = float(y1 + det.y2)

                    all_boxes.append([abs_x1, abs_y1, abs_x2 - abs_x1, abs_y2 - abs_y1])
                    all_scores.append(det.confidence)

        if not all_boxes:
            return []

        # 3. Apply NMS across merged tile bounding boxes
        indices = cv2.dnn.NMSBoxes(all_boxes, all_scores, self.conf_threshold, self.iou_threshold)
        detections: List[DetectionBox] = []

        if len(indices) > 0:
            flat_indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
            for idx in flat_indices:
                bx, by, bw, bh = all_boxes[idx]
                detections.append(
                    DetectionBox(
                        x1=max(0.0, float(bx)),
                        y1=max(0.0, float(by)),
                        x2=min(float(orig_w), float(bx + bw)),
                        y2=min(float(orig_h), float(by + bh)),
                        confidence=all_scores[idx],
                        class_id=0,
                    )
                )

        return detections

    def _detect_pytorch(self, frame: np.ndarray) -> List[DetectionBox]:
        """Inference using PyTorch Ultralytics YOLO engine."""
        orig_h, orig_w = frame.shape[:2]

        results = self.pytorch_model.predict(
            source=frame,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            imgsz=self.input_width,
            device="cpu",
            verbose=False,
        )

        detections: List[DetectionBox] = []
        if not results or results[0].boxes is None or len(results[0].boxes) == 0:
            return detections

        boxes = results[0].boxes
        for box in boxes:
            xyxy = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls_id = int(box.cls[0].cpu().numpy())

            detections.append(
                DetectionBox(
                    x1=max(0.0, float(xyxy[0])),
                    y1=max(0.0, float(xyxy[1])),
                    x2=min(float(orig_w), float(xyxy[2])),
                    y2=min(float(orig_h), float(xyxy[3])),
                    confidence=conf,
                    class_id=cls_id,
                )
            )
        return detections

    def _detect_onnx(self, frame: np.ndarray) -> List[DetectionBox]:
        """Inference using ONNX Runtime CPU engine."""
        orig_h, orig_w = frame.shape[:2]

        # 1. Preprocess Frame
        target_size = self.input_width
        scale = target_size / max(orig_h, orig_w)
        new_w, new_h = int(orig_w * scale), int(orig_h * scale)

        resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        padded = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
        padded[:new_h, :new_w] = resized

        # Normalize HWC BGR -> NCHW RGB [0.0, 1.0]
        blob = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))
        blob = np.expand_dims(blob, axis=0)

        # 2. ONNX Run Inference
        inputs = {self.onnx_session.get_inputs()[0].name: blob}
        outputs = self.onnx_session.run(None, inputs)

        output_tensor = outputs[0]  # Shape (1, 5, 8400) or similar
        if output_tensor.ndim == 3:
            output_tensor = output_tensor[0]  # Shape (5, 8400)

        # Handle transpose if shape is (5, N)
        if output_tensor.shape[0] < output_tensor.shape[1]:
            output_tensor = output_tensor.T  # Shape (N, 5)

        boxes = []
        scores = []

        for row in output_tensor:
            cx, cy, w, h = row[0], row[1], row[2], row[3]
            conf = row[4]

            if conf >= self.conf_threshold:
                # Convert center xy wh to xyxy in padded coordinates
                x1_p = cx - w / 2.0
                y1_p = cy - h / 2.0
                x2_p = cx + w / 2.0
                y2_p = cy + h / 2.0

                # Rescale back to original unpadded frame coordinates
                x1 = max(0.0, min(float(orig_w), x1_p / scale))
                y1 = max(0.0, min(float(orig_h), y1_p / scale))
                x2 = max(0.0, min(float(orig_w), x2_p / scale))
                y2 = max(0.0, min(float(orig_h), y2_p / scale))

                if (x2 - x1) > 5 and (y2 - y1) > 5:
                    boxes.append([x1, y1, x2 - x1, y2 - y1])
                    scores.append(float(conf))

        if not boxes:
            return []

        # 3. Apply NMS (Non-Maximum Suppression)
        indices = cv2.dnn.NMSBoxes(boxes, scores, self.conf_threshold, self.iou_threshold)
        detections: List[DetectionBox] = []

        if len(indices) > 0:
            flat_indices = indices.flatten() if isinstance(indices, np.ndarray) else indices
            for idx in flat_indices:
                x, y, w, h = boxes[idx]
                detections.append(
                    DetectionBox(
                        x1=x,
                        y1=y,
                        x2=x + w,
                        y2=y + h,
                        confidence=scores[idx],
                        class_id=0,
                    )
                )

        return detections
