# 🧠 Hand Gesture & Emotion AI Suite

### 👋 Control your computer with gestures and understand emotions in real time

This project combines **computer vision**, **AI**, and **human–computer interaction** to create a next-gen control and detection suite.  
It lets users interact with their system using **hand gestures** and detects **facial emotions** in real time using deep learning.

---

## 🚀 Features

✅ **Air Mouse** — Control your mouse using hand movement (MediaPipe + PyAutoGUI)  
✅ **Volume Control** — Adjust system volume by pinching your fingers  
✅ **Hand Tracking** — Detect and visualize hand landmarks with precision  
✅ **Emotion AI** — Detect human emotions in real time via webcam  
✅ **Fast & Lightweight** — Optimized frame processing with smoothing and threading  
✅ **Mac M4 Compatible** — Runs efficiently on Apple Silicon  

---

## 🛠️ Tech Stack

| Category | Tools & Frameworks |
|-----------|--------------------|
| **Language** | Python 3.11 |
| **Vision & AI** | OpenCV, MediaPipe, DeepFace |
| **Machine Learning** | TensorFlow (macOS + Metal GPU), Keras |
| **Automation** | PyAutoGUI, Pynput |
| **Visualization** | NumPy, Matplotlib |
| **Device** | macOS (M4 Air / Apple Silicon) |

---

## ⚙️ Installation

### Step 1 — Clone the repository
```bash
git clone https://github.com/sahil24302021/HandGestureController.git
cd HandGestureController
Step 2 — Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate  # For Mac/Linux

Step 3 — Install dependencies
pip install -r requirements.txt


💡 If TensorFlow errors occur, install Apple-optimized versions:

pip install tensorflow-macos tensorflow-metal tf-keras

🧩 Project Modules
1️⃣ hand_tracking.py

Detects hand landmarks using MediaPipe Hands API.

2️⃣ air_mouse.py

Uses finger tracking to control the cursor in real-time.

3️⃣ volume_control.py

Adjusts system volume using the distance between your thumb and index finger.

4️⃣ face_emotion_ai.py

Detects emotions such as happy, sad, angry, surprise, fear, neutral, disgust using DeepFace with advanced smoothing and accuracy optimization.

🧠 How It Works

Captures frames from the webcam

Detects facial landmarks or hand keypoints

Maps movement/emotion to an action (mouse, volume, display)

Smooths results with history-based filtering

Displays real-time overlays with FPS & confidence tracking

🧰 Requirements

macOS / Linux / Windows

Python 3.9+

Webcam

4GB+ RAM (8GB recommended for Emotion AI)

📈 Future Improvements

Voice feedback for detected gestures

Multihand support (dual control)

Cloud-based analytics dashboard

Integration with virtual meeting platforms (Zoom / Meet)

