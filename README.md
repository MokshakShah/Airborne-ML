# Airborne Threat Detection System

A YOLO-based solution for detecting airborne threats using PyTorch. Supports both live webcam feed and video file analysis.

---

## 🛠️ Prerequisites
- [Anaconda/Miniconda](https://www.anaconda.com/download)
- NVIDIA GPU (recommended for CUDA acceleration)
- Python 3.10+

---

## 🚀 Quick Start

### 1. Clone & Navigate to Project
```cmd
cd C:\Users\<your_username>\Documents\volv_my_model
```

---

### 2. Create Conda Environment
```cmd
conda create -n volv-env1 python=3.10
conda activate volv-env1
```

---

### 3. Install Dependencies
```cmd
pip3 install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

---

## 💻 Usage

### Basic Command Structure
```cmd
python python_yolo_detect.py --model <MODEL_PATH> --source <INPUT_SOURCE> [--resolution <WxH>]
```

---

### Live Webcam Detection (1280x720)
```cmd
python python_yolo_detect.py --model my_model1.pt --source webcam --resolution 1280x720
```

**Expected Performance**: ~30 FPS

---

### Video File Detection
```cmd
python yolo_detect.py --model my_model1.pt --source candy_test1.mov
```

---

### Roboflow Workflow Model (Separate from YOLO)
This uses your newly trained Roboflow workflow model in a separate script so existing YOLO logic remains unchanged.

```cmd
pip install inference-sdk
set ROBOFLOW_API_KEY=your_api_key_here
python roboflow_workflow_detect.py --source video1.mp4 --save-video --max-frames 120 --stride 3
```

Outputs are saved in `roboflow_results/`:
- `summary.json`: high-level run summary and counts
- `frame_detections.json`: per-frame detections for reporting
- `*_roboflow_annotated.avi`: annotated demo video (if `--save-video` used)

---

### Far Object Focus + Zoom (Quality Preserving)
Use second-pass ROI zoom to improve detection of distant planes/drones without reducing input quality:

```cmd
python yolo_detect.py --model runs/detect/train3/weights/best.pt --source testvid.mp4 --focus-zoom --preserve-quality --thresh 0.4 --far-object-ratio 0.02 --zoom-factor 2.0
```

**Flags**
- `--focus-zoom`: Enables detect -> focus ROI -> zoom -> detect pipeline.
- `--preserve-quality`: Runs inference on original frame resolution (no pre-inference downscale).
- `--far-object-ratio`: Objects below this bbox/frame area ratio are treated as far/small.
- `--zoom-factor`: Expands ROI around far object before second-pass detection.
