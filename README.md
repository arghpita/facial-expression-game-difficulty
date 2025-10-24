# Facial Expression Recognition for Dynamic Game Difficulty Adjustment

A real-time computer vision system that uses deep learning to recognize facial expressions and automatically adjust game difficulty based on player emotions.

![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.13%2B-orange)
![OpenCV](https://img.shields.io/badge/opencv-4.8%2B-green)
![License](https://img.shields.io/badge/license-Educational-lightgrey)

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Dataset Required](#dataset-required)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Usage Guide](#usage-guide)
- [How It Works](#how-it-works)
- [Troubleshooting](#troubleshooting)
- [Performance Tips](#performance-tips)
- [Known Issues](#known-issues)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgments](#acknowledgments)

---

## Overview

This project demonstrates how facial expression recognition can be used to create adaptive gaming experiences. The system captures video from your webcam, detects your face, classifies your emotional state, and automatically adjusts game difficulty to maintain optimal engagement.

**Key Achievement:** Complete end-to-end ML pipeline from data preprocessing and model training to real-time deployment with an interactive demo game.

### What Makes This Special?

- ✅ **Real-time performance** (30+ FPS on CPU)
- ✅ **No expensive hardware** required (just a webcam)
- ✅ **Fully functional demo game** included
- ✅ **Complete source code** with detailed documentation
- ✅ **Reproducible results** - follow the steps and it works!

---

## Features

### Core Functionality

- **Real-Time Emotion Recognition**: Detects 7 emotions (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral)
- **Adaptive Difficulty System**: Automatically adjusts game parameters based on player emotions
- **Temporal Smoothing**: Stable predictions without jittery behavior
- **Visual Feedback**: Live display of emotions, confidence scores, and difficulty levels
- **Interactive Demo Game**: Playable reaction-time game demonstrating the system

### Technical Features

- Deep CNN trained on FER2013 dataset (60.61% accuracy)
- Haar Cascade face detection
- Multi-level temporal smoothing for stability
- Weighted emotion history for intelligent difficulty calculation
- Comprehensive visualization dashboard

---

## System Requirements

### Minimum Requirements

- **Operating System**: macOS, Linux, or Windows
- **Python**: 3.10 or higher (3.11 recommended)
- **RAM**: 8 GB minimum
- **Webcam**: Any standard webcam (720p recommended)
- **Processor**: Modern multi-core CPU

### Recommended Requirements

- **Python**: 3.11
- **RAM**: 16 GB
- **Webcam**: 1080p with good low-light performance
- **GPU**: NVIDIA GPU with CUDA support (optional, improves training speed)

### Software Dependencies

All dependencies are listed in `requirements.txt`:

```
tensorflow>=2.13.0
opencv-python>=4.8.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
scikit-learn>=1.3.0
pillow>=10.0.0
```
## Dataset Required
   
   This project requires the FER2013 dataset which is not included due to size (300MB).
   
   **Download:**
   1. Go to [Kaggle FER2013](https://www.kaggle.com/datasets/msambare/fer2013)
   2. Download `fer2013.csv`
   3. Place in project root directory

---

## Installation

### Step 1: Clone or Download Project

```bash
# If using Git
git clone <repository-url>
cd expression_recognition

# Or download and extract ZIP file
# cd expression_recognition
```

### Step 2: Install Python 3.11 (if needed)

#### macOS
```bash
# Install Homebrew if not already installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install Python 3.11
brew install python@3.11

# Verify installation
python3.11 --version
```

#### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3.11 python3.11-venv python3.11-dev
```

#### Windows
Download and install Python 3.11 from [python.org](https://www.python.org/downloads/)

### Step 3: Create Virtual Environment

```bash
# Create virtual environment
python3.11 -m venv venv

# Activate virtual environment
# macOS/Linux:
source venv/bin/activate

# Windows:
venv\Scripts\activate

# You should see (venv) in your terminal prompt
```

### Step 4: Install Dependencies

```bash
# Upgrade pip
pip install --upgrade pip

# Install all required packages
pip install -r requirements.txt

# This will take a few minutes
```

### Step 5: Download FER2013 Dataset

**Option A: Kaggle (Recommended)**

1. Create account at [kaggle.com](https://www.kaggle.com)
2. Go to [FER2013 Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
3. Click "Download" button
4. Move `fer2013.csv` to project directory

```bash
# Move downloaded file
mv ~/Downloads/fer2013.csv .
```

**Option B: Alternative Sources**

If you cannot access Kaggle, search for "FER2013 dataset" online. The file should be approximately 300 MB.

### Step 6: Verify Installation

```bash
# Test that all packages are installed correctly
python test_setup.py

# Expected output:
# TensorFlow version: 2.x.x
# OpenCV version: 4.x.x
# Pandas version: 2.x.x
# NumPy version: 1.x.x
# All packages imported successfully!
```

---

## Quick Start

### Option 1: Use Pre-trained Model (Fast)

If you have the pre-trained model files (`best_model.keras`):

```bash
# Activate virtual environment
source venv/bin/activate  # macOS/Linux
# or: venv\Scripts\activate  # Windows

# Run the demo game
python adaptive_game.py
```

### Option 2: Train Model from Scratch (Slow but Complete)

```bash
# Activate virtual environment
source venv/bin/activate

# Train the model (takes 30-60 minutes)
python train_model.py

# Once training completes, run the demo
python adaptive_game.py
```

---

## Project Structure

```
expression_recognition/
│
├── README.md                           # This file
├── requirements.txt                    # Python dependencies
├── fer2013.csv                         # Training dataset (download separately)
│
├── train_model.py                      # Model training script
├── difficulty_controller.py            # Difficulty adjustment logic
├── realtime_detection.py              # Standalone emotion detection
├── integrated_system.py               # Full system integration
├── adaptive_game.py                   # Demo game (MAIN DEMO)
├── test_setup.py                      # Setup verification
│
├── best_model.keras                   # Best trained model (generated)
├── facial_expression_model.keras      # Final model (generated)
├── training_history.png               # Training visualization (generated)
│
├── venv/                              # Virtual environment (created)
└── screenshots/                       # Demo screenshots (optional)
```

---

## Usage Guide

### 1. Training the Model

**If you want to train from scratch:**

```bash
python train_model.py
```

**What happens:**
- Loads FER2013 dataset (~35,000 images)
- Preprocesses images (resize, normalize)
- Trains CNN for up to 50 epochs (with early stopping)
- Saves best model as `best_model.keras`
- Generates training plots in `training_history.png`

**Expected output:**
```
============================================================
Facial Expression Recognition Training
============================================================

[1/6] Loading FER2013 dataset...
Dataset loaded: 35887 samples

[2/6] Preprocessing data...
  Processed 5000/35887 images...
  ...

[3/6] Splitting dataset...
Training set: 25121 samples
Validation set: 5383 samples
Test set: 5383 samples

[4/6] Building CNN model...

[5/6] Training model...
Epoch 1/50
392/392 [==============================] - 45s 115ms/step
...

[6/6] Evaluating model...
Test Accuracy: 60.61%

Training Complete!
```

**Time required:** 30-60 minutes (CPU), 10-20 minutes (GPU)

---

### 2. Testing Individual Components

#### Test A: Difficulty Controller (No webcam needed)

```bash
python difficulty_controller.py
```

Simulates emotion sequences and shows how difficulty adjusts.

**Expected output:**
```
Emotion: Happy (90.0% confidence)
Difficulty: 5.03 (Medium)
Adjustment: +0.030
Recommendation: Player engaged and performing well
```

#### Test B: Real-Time Detection (Requires webcam)

```bash
python realtime_detection.py
```

Opens webcam with emotion detection overlay.

**Features:**
- Face detection with bounding box
- Emotion label above face
- Probability bars for all emotions
- FPS counter

**Controls:**
- `q`: Quit
- `s`: Save screenshot

#### Test C: Integrated System (Requires webcam)

```bash
python integrated_system.py
```

Complete system with emotion detection and difficulty visualization.

**Controls:**
- `q`: Quit
- `s`: Save screenshot
- `r`: Reset difficulty to medium

---

### 3. Playing the Demo Game (MAIN DEMO)

```bash
python adaptive_game.py
```

**Game Objective:**
- Click circular targets before they disappear
- Score +10 points per successful click
- Game over after 10 missed targets

**How Difficulty Adapts:**

| Your Emotion | Game Response |
|--------------|---------------|
| 😠 Angry / 😨 Fear / 😢 Sad | **Easier**: Slower spawns, larger targets, fewer at once |
| 😊 Happy / 😐 Neutral | **Harder**: Faster spawns, smaller targets, more at once |
| 😮 Surprise | Slight increase (engagement indicator) |

**Game Controls:**
- **Mouse Click**: Click targets
- **P**: Pause/Resume
- **R**: Restart game
- **Q**: Quit

**Game Interface:**

```
┌─────────────────────────────┬─────────────┐
│ SCORE: 120   MISSED: 5/10   │  [Webcam]   │
│ TIME: 45s    ACTIVE: 3/4    │             │
│                             │ Emotion:    │
│       ●    ← Targets        │ Happy       │
│                             │ 92.3%       │
│           ●                 │             │
│                             │ DIFFICULTY  │
│    ●                        │ [██████░░]  │
│                             │ 7.2 - Hard  │
│                             │             │
│ Controls: P=Pause  R=Reset │             │
└─────────────────────────────┴─────────────┘
```

**Tips for Best Results:**
1. Position yourself 2-3 feet from camera
2. Ensure good lighting (face your light source)
3. Look directly at the camera
4. Make exaggerated expressions to test the system
5. Try staying frustrated for 10+ seconds to see difficulty drop

---

## How It Works

### System Pipeline

```
1. WEBCAM INPUT
   ↓
2. FACE DETECTION (Haar Cascade)
   ↓
3. PREPROCESSING (Resize to 48×48, normalize)
   ↓
4. CNN MODEL (Emotion classification)
   ↓
5. TEMPORAL SMOOTHING (Average last 10 frames)
   ↓
6. DIFFICULTY CONTROLLER (Calculate adjustment)
   ↓
7. GAME PARAMETERS (Update spawn rate, size, etc.)
   ↓
8. VISUAL FEEDBACK (Display everything)
```

### Model Architecture

```
Input (48×48×1 grayscale image)
    ↓
[Convolutional Block 1]
  - Conv2D(32 filters) → BatchNorm → ReLU
  - Conv2D(32 filters) → BatchNorm → ReLU
  - MaxPool(2×2) → Dropout(0.25)
    ↓
[Convolutional Block 2]
  - Conv2D(64 filters) → BatchNorm → ReLU
  - Conv2D(64 filters) → BatchNorm → ReLU
  - MaxPool(2×2) → Dropout(0.25)
    ↓
[Convolutional Block 3]
  - Conv2D(128 filters) → BatchNorm → ReLU
  - Conv2D(128 filters) → BatchNorm → ReLU
  - MaxPool(2×2) → Dropout(0.25)
    ↓
[Dense Layers]
  - Flatten
  - Dense(256) → BatchNorm → Dropout(0.5)
  - Dense(128) → BatchNorm → Dropout(0.5)
  - Dense(7, softmax)
    ↓
Output (7-class probability distribution)
```

### Difficulty Adjustment Algorithm

```python
# Emotion weights (how each emotion affects difficulty)
weights = {
    'Angry': -0.8,    # Strong decrease
    'Fear': -0.7,     # Strong decrease
    'Sad': -0.6,      # Moderate decrease
    'Disgust': -0.5,  # Moderate decrease
    'Happy': +0.3,    # Slight increase
    'Surprise': +0.2, # Slight increase
    'Neutral': +0.1   # Maintain/slight increase
}

# Calculate weighted adjustment from emotion history
adjustment = Σ(weight[emotion] × confidence × recency) / Σ(confidence × recency)

# Apply adjustment rate for smoothness
adjustment *= 0.15

# Update difficulty within bounds [1, 10]
new_difficulty = clamp(current_difficulty + adjustment, 1, 10)
```

---

## Troubleshooting

### Issue 1: "command not found: python3.11"

**Solution:**
```bash
# Check what Python versions you have
python3 --version
ls /usr/local/bin/python*

# Use whatever version is 3.10+
python3.10 -m venv venv
```

### Issue 2: "ModuleNotFoundError: No module named 'XXX'"

**Solution:**
```bash
# Make sure virtual environment is activated
source venv/bin/activate  # You should see (venv) in prompt

# Reinstall requirements
pip install -r requirements.txt
```

### Issue 3: Webcam not opening

**Solution:**

**macOS:**
- System Preferences → Security & Privacy → Camera
- Allow Terminal/Python to access camera

**Linux:**
```bash
# Check camera devices
ls /dev/video*

# Try different camera index
# Edit adaptive_game.py line: cap = cv2.VideoCapture(1)  # Try 1 instead of 0
```

**Windows:**
- Check Camera privacy settings
- Ensure no other app is using camera

### Issue 4: Low FPS / Slow performance

**Solution:**
```bash
# Reduce frame processing
# In adaptive_game.py, add frame skipping:
if frame_count % 2 == 0:  # Process every other frame
    emotion, confidence = detect_emotion(frame)
```

### Issue 5: Face not detected

**Causes & Solutions:**
- **Poor lighting**: Face a light source, avoid backlighting
- **Profile view**: Face camera directly
- **Too close/far**: Stay 2-3 feet from camera
- **Occlusions**: Remove glasses, masks, or hands from face

### Issue 6: Training fails with memory error

**Solution:**
```bash
# Reduce batch size in train_model.py
# Line ~120: Change batch_size=64 to batch_size=32
```

### Issue 7: "AttributeError: module 'cv2' has no attribute 'FONT_HERSHEY_BOLD'"

**Solution:**
```bash
# Replace FONT_HERSHEY_BOLD with FONT_HERSHEY_DUPLEX
sed -i '' 's/FONT_HERSHEY_BOLD/FONT_HERSHEY_DUPLEX/g' adaptive_game.py
```

---

## Performance Tips

### For Better Accuracy

1. **Good Lighting**: 
   - Face light source
   - Avoid shadows on face
   - Natural light works best

2. **Camera Positioning**:
   - 2-3 feet from camera
   - Camera at eye level
   - Face camera directly

3. **Clear Expressions**:
   - Exaggerate emotions slightly
   - Hold expressions for 2-3 seconds
   - Avoid rapid expression changes

### For Better Performance

1. **Close Unnecessary Apps**: Free up CPU/RAM

2. **Reduce Resolution**:
```python
# In adaptive_game.py
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Instead of 1280
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Instead of 720
```

3. **Use GPU** (if available):
```bash
# Install GPU-enabled TensorFlow
pip install tensorflow[and-cuda]
```

---

## Known Issues

1. **Accuracy Limitation**: 60.61% accuracy is standard for FER2013 dataset due to:
   - Low resolution (48×48 pixels)
   - Inconsistent labeling in dataset
   - Limited facial detail

2. **Lighting Sensitivity**: Performance degrades in poor lighting

3. **Single Player Only**: System tracks one face at a time

4. **Cultural Differences**: Model trained on specific dataset may not generalize to all cultural expression styles

5. **Expression Intensity**: Binary classification (emotion present/absent) rather than intensity measurement

---

## Future Improvements

### Short-term
- [ ] Add GPU acceleration for faster processing
- [ ] Implement better face detector (MTCNN, RetinaFace)
- [ ] Add emotion intensity measurement
- [ ] Support multiple players simultaneously

### Long-term
- [ ] Train on better datasets (AffectNet, RAF-DB)
- [ ] Implement transfer learning with pre-trained models
- [ ] Add personalization (learn individual baselines)
- [ ] Integrate with actual game engines (Unity, Unreal)
- [ ] Mobile deployment (iOS, Android)

---

## Contributing

This is an educational project. Suggestions and improvements welcome!

**To contribute:**
1. Fork the repository
2. Create feature branch
3. Make changes with clear comments
4. Test thoroughly
5. Submit pull request with description

---

## License

This project is for educational purposes. 

**Dataset License**: FER2013 dataset has its own terms of use.

**Code License**: Open for educational use. Not for commercial use without permission.

---

## Acknowledgments

- **FER2013 Dataset**: Goodfellow et al., 2013 - ICML Challenges in Representation Learning
- **TensorFlow/Keras**: Deep learning framework
- **OpenCV**: Computer vision library
- **Course**: ALY6080 / CMSC 498L with Professor Kenneth Parker
- **Community**: Stack Overflow, GitHub, Kaggle communities

---

## Citation

If you use this project in your work, please cite:

```
[Your Name] (2025). Facial Expression Recognition for Dynamic Game Difficulty Adjustment.
Educational project for ALY6080/CMSC 498L. [University Name].
```

---

## Contact & Support

**Questions?** Check:
1. This README (you're here!)
2. Detailed report: `FacialExpressionRecognition_[YourName]_Report.md`
3. Code comments in Python files

**Still stuck?** 
- Review troubleshooting section above
- Check system requirements
- Verify all installation steps completed

---

## Quick Command Reference

```bash
# Setup
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train model
python train_model.py

# Test components
python test_setup.py
python difficulty_controller.py
python realtime_detection.py

# Run demos
python integrated_system.py    # Full system
python adaptive_game.py        # Demo game (MAIN)

# Deactivate environment
deactivate
```

---

**Ready to start?** Follow the [Installation](#installation) section above!

**Want to see it in action?** Jump to [Quick Start](#quick-start)!

**Having issues?** Check [Troubleshooting](#troubleshooting)!

---

*Last Updated: October 2025*  
*Version: 1.0*  
*Status: ✅ Complete and Working*
