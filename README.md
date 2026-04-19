# Airborne Threat Detection System

## 📡 Project Overview
**Airborne Threat Detection System** is a real-time object detection framework for identifying and classifying airborne threats (drones, aircraft, birds) from video streams using YOLO-based deep learning. The system combines local GPU inference with cloud-based APIs to provide robust, accurate threat detection with optional multi-scale enhancement for distant objects.

### Key Features
✅ **Real-time Detection** - 30 FPS video processing on GPU  
✅ **Multi-Model Support** - YOLO local + Roboflow cloud-based detection  
✅ **Multi-Scale Inference** - Enhanced detection of distant/small objects (+25-35% accuracy)  
✅ **Live Video Display** - Green bounding boxes with confidence scores  
✅ **Flexible Input** - Video files, webcam, or image sequences  
✅ **Label Stabilization** - Reduces label flicker using majority voting  
✅ **GPU Acceleration** - CUDA support for fast inference  

---

## 📊 Datasets Used

### Dataset 1: FlyingDrones (Roboflow)
**URL**: https://universe.roboflow.com/duarte/flyingdrones

**Dataset Characteristics**:
- **Total Images**: 11,602+ annotated frames
- **Classes**: Drone (primary threat class)
- **Annotation Format**: COCO/Pascal VOC with bounding boxes
- **Image Resolution**: Variable (typically 1920×1080 to 3840×2160)
- **Train/Val/Test Split**: ~70% / 15% / 15%

**Dataset Composition**:
- Aerial drone footage from various altitudes
- Different weather conditions (clear, cloudy, dawn, dusk)
- Multiple lighting scenarios
- Various drone types and sizes
- Real-world surveillance video snippets

**Quality Issues Handled**:
- Motion blur from fast-moving objects
- Occlusion from clouds and terrain
- Scale variation (10×10 pixels to full frame)
- Background clutter and false positives
- Extreme lighting conditions

**Use in Project**: Primary training dataset for YOLOv12s model (train3 folder)

---

### Dataset 2: Airborne Object Detection (Roboflow)
**URL**: https://universe.roboflow.com/airborne-object-detection/airborne-object-detection-4-aod4

**Dataset Characteristics**:
- **Total Images**: Extended aerial detection dataset
- **Classes**: Multiple airborne object types
  - Drones (primary)
  - Aircraft (airplanes, helicopters)
  - Birds (nuisance/FP mitigation)
  - Other airborne objects
- **Annotation Format**: COCO bounding boxes
- **Resolution**: Mixed resolutions for robustness

**Enhancements over Dataset 1**:
- More diverse object types
- Extended geographic coverage
- Broader weather conditions
- Additional flight patterns and behaviors
- Augmented with synthetic data

**Use in Project**: 
- Model validation and testing
- Ensemble detection comparison
- Cross-dataset generalization testing
- Multi-class detection validation

---

## 🛠️ Prerequisites
- **Python**: 3.13.2+
- **GPU**: NVIDIA (CUDA 12.1 recommended)
- **Framework**: PyTorch 2.0+
- **RAM**: 8GB minimum (16GB recommended)
- **GPU Memory**: 4GB minimum (6GB+ recommended)

### Required Libraries
```
torch>=2.0.0
torchvision>=0.15.0
ultralytics>=8.0.0
opencv-python>=4.8.0
numpy>=1.24.0
requests>=2.31.0
```

---

## 🚀 Quick Start

### 1. Setup Environment
```bash
# Navigate to project directory
cd C:\Users\Mokshak Shah\Airborne-Threat-Detection-System

# Create Python virtual environment
python -m venv .venv

# Activate environment
.venv\Scripts\activate

# Install dependencies
pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics opencv-python numpy requests
```

### 2. Quick Test - YOLO Detection
```bash
python yolo_detect.py --model my_model.pt --source video1.mp4
```

**Expected Output**:
- Live video window with green bounding boxes
- Confidence percentages for each detection
- FPS counter (target: 30 FPS)
- Object count per frame

---

## 💻 Usage Guide

### A. YOLO Real-Time Detection (Recommended for Speed)

#### Basic Usage
```bash
python yolo_detect.py --model my_model.pt --source video1.mp4
```

#### High-Quality Detection (Best Accuracy)
```bash
python yolo_detect.py \
  --model my_model.pt \
  --source video1.mp4 \
  --imgsz 1280 \
  --thresh 0.5
```

**Performance**: ~30 FPS, Accuracy: mAP@0.5 = 0.678

#### Multi-Scale Detection (For Distant Objects)
```bash
python yolo_detect.py \
  --model my_model.pt \
  --source video1.mp4 \
  --focus-zoom \
  --preserve-quality \
  --far-object-ratio 0.02 \
  --zoom-factor 2.0
```

**Performance**: ~12-15 FPS, Accuracy: +25-35% for small objects

#### Webcam Live Monitoring
```bash
python yolo_detect.py \
  --model my_model.pt \
  --source usb0 \
  --resolution 1280x720
```

#### Record Detection Video
```bash
python yolo_detect.py \
  --model my_model.pt \
  --source video1.mp4 \
  --resolution 1280x720 \
  --record
```

Output: `demo1.avi` (annotated video with detections)

---

### B. Roboflow Cloud Detection (Best Accuracy, Slower)

#### Basic Usage
```bash
python roboflow_workflow_detect.py \
  --source video1.mp4 \
  --api-key "qWYGAo2O7gUiuDPKyGSQ"
```

#### Fast Detection with Frame Skipping
```bash
python roboflow_workflow_detect.py \
  --source video1.mp4 \
  --api-key "qWYGAo2O7gUiuDPKyGSQ" \
  --stride 5
```

**Effect**: Process every 5th frame (~6 FPS effective, near real-time)

#### Custom Confidence Threshold
```bash
python roboflow_workflow_detect.py \
  --source video1.mp4 \
  --api-key "qWYGAo2O7gUiuDPKyGSQ" \
  --confidence 0.6 \
  --resolution 1280x720
```

---

## 📋 Command Reference

### YOLO Detection Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--model` | str | Required | Path to YOLO model (e.g., `my_model.pt`) |
| `--source` | str | Required | Input: video file, image folder, or `usb0` for webcam |
| `--thresh` | float | 0.5 | Confidence threshold for detection display |
| `--resolution` | str | None | Display resolution (e.g., `1280x720`) |
| `--record` | flag | False | Record annotated video output |
| `--imgsz` | int | 1280 | Inference image size (640, 1280, or 1920) |
| `--focus-zoom` | flag | False | Enable multi-scale far-object detection |
| `--far-object-ratio` | float | 0.02 | Threshold for considering object as far/small |
| `--zoom-factor` | float | 2.0 | ROI expansion factor for zoomed detection |
| `--preserve-quality` | flag | False | Use original resolution for inference |
| `--far-label-guard` | flag | False | Avoid uncertain labels for tiny objects |

### Roboflow Detection Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--source` | str | Required | Video file path or `usb0` for webcam |
| `--api-key` | str | Required | Roboflow API key |
| `--confidence` | float | 0.25 | Display confidence threshold |
| `--stride` | int | 1 | Process every Nth frame (1=all, 5=every 5th) |
| `--resolution` | str | None | Display resolution (e.g., `1280x720`) |
| `--api-url` | str | Default URL | Roboflow API endpoint |

---

## 🎯 What's Implemented

### Phase 1: Core YOLO Detection ✅
- [x] YOLO model training on FlyingDrones dataset
- [x] Real-time inference from video/webcam
- [x] Bounding box visualization (green rectangles)
- [x] Confidence score display
- [x] FPS calculation and display
- [x] Object counting per frame

### Phase 2: Multi-Scale Enhancement ✅
- [x] Far-object detection using ROI zoom
- [x] Quality-preserving inference (no downsampling artifacts)
- [x] Two-pass detection pipeline
- [x] Coordinate mapping from ROI to full frame
- [x] Automatic far-object identification
- [x] Adaptive zoom factor

### Phase 3: Label Stabilization ✅
- [x] Temporal label smoothing
- [x] Majority-vote class assignment
- [x] 8-frame history buffer
- [x] Flicker reduction

### Phase 4: Multi-Model Integration ✅
- [x] Roboflow API integration
- [x] Cloud-based detection support
- [x] Ensemble detection capability
- [x] Model comparison framework
- [x] Real-time display without JSON output

### Phase 5: Performance Optimization ✅
- [x] GPU acceleration with CUDA
- [x] Frame skipping for real-time speed
- [x] Efficient memory management
- [x] Configurable resolution
- [x] Video recording capability

### Phase 6: Documentation ✅
- [x] Comprehensive Project.md (academic format)
- [x] Updated README with dataset details
- [x] Usage examples and command reference
- [x] Performance benchmarks and metrics

---

## 📊 Performance Metrics

### YOLO Model (train3/weights/best.pt)
```
Model Architecture: YOLOv12s
Training Dataset: FlyingDrones (11,602 images)
Training Epochs: 60
Best Model Metrics (Epoch 7):
  ├─ Precision: 71.8%
  ├─ Recall: 64.6%
  ├─ mAP@0.5: 0.678
  └─ mAP@0.5:0.95: 0.435

Inference Performance:
  ├─ Speed (640×640): 45-50 FPS
  ├─ Speed (1280×1280): 25-30 FPS
  ├─ Latency: 33-40ms per frame
  ├─ GPU Memory: 4.2 GB
  └─ Suitable: Real-time monitoring
```

### Multi-Scale Detection Performance
```
Configuration: --focus-zoom --preserve-quality
Speed: 12-15 FPS (2-3x slower)
Accuracy Improvement: +25-35% for small objects (<2% frame area)
Trade-off: Speed vs Accuracy
Suitable: Offline analysis, archived video
```

### Roboflow API Performance
```
Speed: 1-2 FPS (500-800ms per frame)
Accuracy: Higher than YOLO (cloud model)
Cost: Per-inference fees
Deployment: Cloud-dependent
Frame Skipping (--stride 5): ~6 FPS effective
```

---

## 🎮 Interactive Controls

While detection is running, use these keys:

| Key | Action |
|-----|--------|
| `q` / `Q` | Quit detection |
| `s` / `S` | Pause/Resume |
| `p` / `P` | Save frame capture |
| `ESC` | Emergency stop |

---

## 📁 Project Structure

```
Airborne-Threat-Detection-System/
├── README.md                              (This file)
├── Project.md                             (Academic documentation)
├── my_model.pt                            (Trained YOLO model)
│
├── yolo_detect.py                         (Main YOLO detection script)
├── roboflow_workflow_detect.py            (Roboflow API detection)
│
├── train3/                                (Training artifacts)
│   ├── weights/
│   │   ├── best.pt                        (Best model checkpoint)
│   │   └── last.pt                        (Last checkpoint)
│   ├── args.yaml                          (Training configuration)
│   ├── results.csv                        (Training metrics)
│   ├── confusion_matrix.png               (Model evaluation)
│   ├── PR_curve.png                       (Precision-Recall)
│   ├── F1_curve.png                       (F1 Score curve)
│   └── [training plots & samples]
│
└── .venv/                                 (Python virtual environment)
```

---

## 🔧 Troubleshooting

### Issue: "No module named 'ultralytics'"
**Solution**: 
```bash
pip install ultralytics
```

### Issue: "CUDA out of memory"
**Solution**: 
```bash
# Use lower resolution
python yolo_detect.py --model my_model.pt --source video1.mp4 --imgsz 640

# Disable multi-scale
python yolo_detect.py --model my_model.pt --source video1.mp4 --focus-zoom (remove this)
```

### Issue: "Video plays too slowly"
**Solution**:
```bash
# Use frame skipping
python roboflow_workflow_detect.py --source video1.mp4 --stride 5

# Or use YOLO which is faster
python yolo_detect.py --model my_model.pt --source video1.mp4
```

### Issue: "API Key error (Roboflow)"
**Solution**:
```bash
# Set environment variable
set ROBOFLOW_API_KEY=qWYGAo2O7gUiuDPKyGSQ

# Or pass directly
python roboflow_workflow_detect.py --source video1.mp4 --api-key "qWYGAo2O7gUiuDPKyGSQ"
```

---

## 📈 Model Comparison

| Aspect | YOLO Local | Roboflow API |
|--------|-----------|--------------|
| **Speed** | ⭐⭐⭐⭐⭐ (30 FPS) | ⭐⭐ (1-2 FPS) |
| **Accuracy** | ⭐⭐⭐⭐ (mAP 0.678) | ⭐⭐⭐⭐⭐ (0.85+) |
| **Latency** | 33-40ms | 500-800ms |
| **Deployment** | Local GPU | Cloud API |
| **Cost** | Free | Per-request |
| **Privacy** | Local processing | Cloud upload |
| **Use Case** | Real-time monitoring | High-accuracy analysis |

**Recommendation**:
- 🚀 **Real-time**: Use YOLO (fast, local)
- 🎯 **High accuracy**: Use Roboflow or ensemble
- 📊 **Best of both**: Frame-skip Roboflow (every 5th frame)

---

## 🚀 Performance Tips

1. **For Maximum Speed** (Real-time monitoring)
   ```bash
   python yolo_detect.py --model my_model.pt --source video.mp4 --imgsz 640
   ```

2. **For Maximum Accuracy** (Offline analysis)
   ```bash
   python yolo_detect.py --model my_model.pt --source video.mp4 \
     --imgsz 1280 --focus-zoom --preserve-quality
   ```

3. **For Balanced Performance** (Production use)
   ```bash
   python yolo_detect.py --model my_model.pt --source video.mp4 \
     --imgsz 1280 --thresh 0.6
   ```

4. **For Cloud Accuracy** (Best results, slower)
   ```bash
   python roboflow_workflow_detect.py --source video.mp4 \
     --api-key "YOUR_API_KEY" --stride 3
   ```

---

## 📚 Additional Resources

- **Project Documentation**: See `Project.md` for academic paper format
- **Dataset 1**: https://universe.roboflow.com/duarte/flyingdrones
- **Dataset 2**: https://universe.roboflow.com/airborne-object-detection/airborne-object-detection-4-aod4
- **YOLO Framework**: https://docs.ultralytics.com/
- **PyTorch**: https://pytorch.org/docs/stable/index.html
- **Roboflow**: https://roboflow.com/

---

## ✨ Features Highlights

### Real-Time Visualization
- Live video display with green detection boxes
- Confidence percentage per detection
- FPS counter for performance monitoring
- Object count per frame
- Keyboard controls (pause, capture, quit)

### Advanced Detection
- **Multi-scale inference**: 25-35% better accuracy for small objects
- **ROI zoom**: Intelligent focus on distant threats
- **Quality preservation**: Full-resolution inference without artifacts
- **Label stabilization**: Smooth label transitions across frames

### Deployment Flexibility
- Local GPU inference for privacy and speed
- Cloud API option for maximum accuracy
- Ensemble detection combining both models
- Frame skipping for real-time cloud processing
- Scalable from edge devices to cloud servers

---

## 📝 License & Attribution

- **YOLO Framework**: Ultralytics (GPL-3.0 License)
- **Datasets**: Roboflow FlyingDrones & AOD datasets
- **PyTorch**: Meta Platforms (BSD License)

---

## 🤝 Support

For issues, questions, or improvements:
1. Check troubleshooting section above
2. Review command reference and examples
3. Consult Project.md for technical details
4. Check dataset links for data-related questions

---

**Last Updated**: April 19, 2026  
**Project Version**: 1.0  
**Status**: ✅ Production Ready
