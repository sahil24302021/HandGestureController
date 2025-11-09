# Emotion Pro v12 — Fast, Accurate, Animated Confidence Gauge (Ensemble)
# - Real-time ensemble of DeepFace + FER
# - Preloaded DeepFace emotion model, threaded inference
# - Fast preprocessing, smoothing, and animated circular confidence meter
# Paste into your face_emotion_ai.py (replace existing content)

import cv2
import time
import threading
import numpy as np
import os
from collections import deque
from deepface import DeepFace
from fer import FER
import math

# ---------------- CONFIG ----------------
EMOTION_COLORS = {
    "happy": (0, 255, 0),
    "sad": (255, 0, 0),
    "angry": (0, 0, 255),
    "surprise": (255, 255, 0),
    "neutral": (200, 200, 200),
    "fear": (255, 128, 0),
    "disgust": (128, 0, 128)
}
EMOJIS = {
    "happy": "😄", "sad": "😢", "angry": "😡",
    "surprise": "😲", "neutral": "😐", "fear": "😨", "disgust": "🤢"
}

SMOOTH_WINDOW = 8                 # smoothing window (frames)
emotion_history = deque(maxlen=SMOOTH_WINDOW)
frame_buffer = None
lock = threading.Lock()
running = True

# Speed-related environment settings (use sensible defaults)
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_USE_LEGACY_KERAS"] = "1"


# ---------------- INITIALIZE CAM + MODELS ----------------
cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 720)

print("⚡ Emotion Pro v12 — Animated Gauge (press q to quit)")

# Preload DeepFace emotion model (so inference is fast)
print("Loading DeepFace emotion model...")
DF_MODEL = DeepFace.build_model("Emotion")
print("Loading FER model...")
# FER with mtcnn=False is faster; DeepFace(opencv) handles detection for accuracy
fer_detector = FER(mtcnn=False)
print("✅ Models ready")

# ---------------- HELPERS ----------------
def preprocess(img, target_w=640):
    """Resize to target width & CLAHE equalization for stable inference."""
    h, w = img.shape[:2]
    if w > target_w:
        scale = target_w / float(w)
        img = cv2.resize(img, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(2.0, (8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

def combine_probs(d1, d2, w1=0.65, w2=0.35):
    """Weighted merge and normalize two probability dicts."""
    out = {}
    keys = set(d1.keys()) | set(d2.keys())
    for k in keys:
        v1 = d1.get(k, 0)
        v2 = d2.get(k, 0)
        out[k] = v1 * w1 + v2 * w2
    s = sum(out.values())
    if s > 0:
        out = {k: v / s for k, v in out.items()}
    return out

def draw_confidence_gauge(panel, center, radius, confidence, color):
    """
    Draws a circular gauge on 'panel'.
    - confidence: 0..100
    - color: (b,g,r)
    Adds multi-layer glow and animated pulse.
    """
    # background circle
    cv2.circle(panel, center, radius, (30, 30, 30), -1)
    # animated pulse intensity (0.6..1.2)
    t = time.time()
    pulse = 0.9 + 0.15 * math.sin(t * 2.5)
    # outer glow (translucent)
    overlay = panel.copy()
    glow_color = tuple(min(255, int(c * 0.6)) for c in color)
    cv2.circle(overlay, center, int(radius * 1.15), glow_color, -1)
    cv2.addWeighted(overlay, 0.05 * pulse, panel, 0.95, 0, panel)

    # arc for confidence
    start_angle = -90
    end_angle = int(start_angle + (confidence / 100.0) * 360)
    # draw thick arc (base)
    cv2.ellipse(panel, center, (radius, radius), 0, start_angle, end_angle, color, 14)
    # draw inner bright arc for highlight
    highlight = tuple(min(255, int(c * 1.1)) for c in color)
    cv2.ellipse(panel, center, (radius - 10, radius - 10), 0, start_angle, end_angle, highlight, 6)

    # text (percentage)
    txt = f"{int(confidence)}%"
    cv2.putText(panel, txt, (center[0] - 35, center[1] + 10),
                cv2.FONT_HERSHEY_DUPLEX, 1.2, (225, 225, 225), 2)

# ---------------- ANALYSIS THREAD ----------------
def analyze_thread():
    global frame_buffer, running
    while running:
        frame_local = None
        with lock:
            if frame_buffer is not None:
                frame_local = frame_buffer.copy()
        if frame_local is None:
            time.sleep(0.01)
            continue

        proc = preprocess(frame_local)
        try:
            # DeepFace inference (use OpenCV detector backend for reliable face detection)
            res = DeepFace.analyze(
                proc,
                actions=['emotion'],
                models={"emotion": DF_MODEL},
                detector_backend='opencv',   # robust detection
                enforce_detection=False,
                silent=True
            )
            # unify format (sometimes returns list)
            if isinstance(res, list) and len(res) > 0:
                res = res[0]
            df_emotions = res.get("emotion", {})
            total = sum(df_emotions.values()) or 1.0
            df_emotions = {k: v / total for k, v in df_emotions.items()}
        except Exception:
            df_emotions = {}

        # FER inference (fast, no mtcnn to save time)
        try:
            fer_res = fer_detector.detect_emotions(proc)
            if fer_res:
                fer_em = fer_res[0].get("emotions", {})
                total2 = sum(fer_em.values()) or 1.0
                fer_em = {k: v / total2 for k, v in fer_em.items()}
            else:
                fer_em = {}
        except Exception:
            fer_em = {}

        # Ensemble: prefer DeepFace if it returned something, else rely on FER
        if df_emotions:
            combined = combine_probs(df_emotions, fer_em, w1=0.7, w2=0.3)
        elif fer_em:
            combined = fer_em
        else:
            combined = {}

        if combined:
            # append combined probability vector to history (smoothing on probabilities)
            emotion_history.append(combined)

        # small sleep to avoid maxing CPU (keeps high throughput)
        time.sleep(0.02)

# start analysis thread
threading.Thread(target=analyze_thread, daemon=True).start()

# ---------------- MAIN UI LOOP ----------------
fps_count = 0
fps_timer = time.time()
fps = 0

try:
    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        with lock:
            frame_buffer = frame

        h, w = frame.shape[:2]

        # Build right-side panel
        panel_w = 380
        panel = np.zeros((h, panel_w, 3), dtype=np.uint8)
        panel[:] = (12, 12, 18)  # dark background

        # FPS
        fps_count += 1
        if time.time() - fps_timer >= 1.0:
            fps = fps_count
            fps_count = 0
            fps_timer = time.time()
        cv2.putText(panel, f"FPS: {fps}", (18, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 220, 120), 2)

        # Average probabilities over history (probability-level smoothing)
        avg_probs = {}
        if len(emotion_history) > 0:
            # sum then normalize
            for pvec in emotion_history:
                for k, v in pvec.items():
                    avg_probs[k] = avg_probs.get(k, 0.0) + v
            for k in avg_probs:
                avg_probs[k] /= len(emotion_history)

        if avg_probs:
            # main emotion and confidence
            main = max(avg_probs, key=avg_probs.get)
            conf = float(avg_probs[main]) * 100.0
            color = EMOTION_COLORS.get(main, (255, 255, 255))
            emoji = EMOJIS.get(main, "🙂")

            # Header block
            cv2.rectangle(panel, (0, 60), (panel_w, 140), color, -1)
            cv2.putText(panel, f"{main.upper()} {emoji}", (18, 115),
                        cv2.FONT_HERSHEY_DUPLEX, 1.2, (0, 0, 0), 3)
            cv2.putText(panel, f"{int(conf)}%", (panel_w - 70, 115),
                        cv2.FONT_HERSHEY_DUPLEX, 0.9, (0, 0, 0), 2)

            # Draw animated circular confidence gauge
            gauge_center = (panel_w // 2, 220)
            gauge_radius = 60
            draw_confidence_gauge(panel, gauge_center, gauge_radius, conf, color)

            # Draw sorted bars
            y0 = 320
            for emo, score in sorted(avg_probs.items(), key=lambda x: x[1], reverse=True):
                bar_len = int(score * (panel_w - 70))
                bar_color = EMOTION_COLORS.get(emo, (200, 200, 200))
                cv2.rectangle(panel, (30, y0), (30 + bar_len, y0 + 22), bar_color, -1)
                cv2.rectangle(panel, (30, y0), (panel_w - 30, y0 + 22), (60, 60, 60), 1)
                cv2.putText(panel, f"{emo}: {int(score*100)}%", (35, y0 + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
                y0 += 32
        else:
            cv2.putText(panel, "Detecting...", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 60, 200), 2)

        # Combine and show
        combined = np.hstack((frame, panel))
        cv2.imshow("⚡ Emotion Pro v12 — Animated Confidence Gauge", combined)

        # exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        # maintain minimal frame pacing (keeps UI smooth)
        elapsed = time.time() - start
        if elapsed < 0.01:
            time.sleep(0.01 - elapsed)

except KeyboardInterrupt:
    pass

# ---------------- CLEANUP ----------------
running = False
time.sleep(0.05)
cap.release()
cv2.destroyAllWindows()
