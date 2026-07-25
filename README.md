##**Automatic License Plate Recognition (ALPR)**

A high-speed, CPU-optimized Automatic License Plate Recognition (ALPR) system focused on real-time video playback, accurate long-distance detection, blur filtering, OCR caching, and annotated video generation.

---

## Key Features

* **Multi-Threaded Architecture**: 3 dedicated workers (Video Reader, Inference Worker, Display & Writer) with thread-safe queues.
* **ONNX CPU Acceleration**: Automatically converts PyTorch models to ONNX and uses ONNX Runtime for 2x–3x faster CPU execution with PyTorch fallback.
* **ROI Localized Search**: Tracks bounding boxes across frames to inspect small regions of interest, skipping full-frame inference on intermediate frames.
* **Blur Metric Filtering**: Uses Laplacian variance to evaluate image focus, skipping blurry crops and waiting for sharp frames before invoking OCR.
* **OCR Caching & Track Stability**: Requires consecutive frame track stability before running PaddleOCR once and caching text per track ID.
* **Adaptive Frame Skipping**: Dynamically scales skip rate based on real-time CPU load to ensure smooth playback.
* **Multiple Trackers**: Supports **Centroid Tracker** (lightweight default), **SORT**, and **ByteTrack**.
* **Performance Modes**: `ultra_fast`, `balanced`, and `high_accuracy`.
* **Benchmark Reporting**: Detailed metrics via `python run.py --benchmark`.

---

## Directory Structure

```
ALPR/
├── models/
│   ├── pretrained/
│   ├── best_plate_detector.pt
│   └── best_plate_detector.onnx
├── videos/
│   ├── input/
│   └── output/
├── src/
│   ├── detection/
│   │   ├── plate_detector.py
│   │   └── export_onnx.py
│   ├── tracking/
│   │   ├── centroid_tracker.py
│   │   ├── sort_tracker.py
│   │   ├── bytetrack_tracker.py
│   │   └── tracker.py
│   ├── ocr/
│   │   ├── blur_detector.py
│   │   └── ocr_engine.py
│   ├── optimization/
│   │   └── adaptive_skip.py
│   ├── pipeline/
│   │   ├── roi_detector.py
│   │   └── video_pipeline.py
│   └── utils/
│       ├── drawing.py
│       ├── benchmark.py
│       └── video.py
├── config.py
├── run.py
├── requirements.txt
└── README.md
```

---

## Installation & Setup

1. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

2. **Verify System Setup**:
   ```bash
   python run.py --check-only
   ```

---

## Usage

### Run on Default / Interactive Video Selection
```bash
python run.py
```

### Process Specific Video File
```bash
python run.py --video videos/input/1.mp4
```

### Benchmark Mode
Measure FPS, detection latency, OCR latency, CPU %, and memory usage:
```bash
python run.py --video videos/input/1.mp4 --benchmark
```

### Select Performance Mode
* **Ultra Fast**: Lower resolution, aggressive skip rate, Centroid Tracker.
  ```bash
  python run.py --mode ultra_fast
  ```
* **Balanced** (Default):
  ```bash
  python run.py --mode balanced
  ```
* **High Accuracy**: Higher resolution, minimal frame skip, SORT tracker.
  ```bash
  python run.py --mode high_accuracy
  ```

---

## Final Output & Summary

* **Annotated Video**: Saved to `videos/output/<video_name>_annotated.mp4`.
* **Summary Report**: Printed to console upon completion:
  - Video Name
  - Processing Time
  - Average FPS
  - Frames Processed
  - Unique Plates Detected (with confidence score and Track ID)
=======
# automatic_license_plate_recognition
Automatic License Plate Recognition (ALPR) system that detects license plates from videos, extracts plate numbers using OCR, and generates annotated output videos.
>>>>>>> 3845f6de49a67a348b3b13727e1db151b63760c1
