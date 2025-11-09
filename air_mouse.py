import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import math
import time
from pynput.mouse import Controller, Button

# ==========================
# CONFIGURATION
# ==========================
SMOOTHING = 0.7
CLICK_COOLDOWN = 0.3  # seconds
PINCH_THRESHOLD = 40  # distance for click
SCROLL_SENSITIVITY = 3  # scroll speed

# ==========================
# INITIALIZATION
# ==========================
mp_hands = mp.solutions.hands
hands_detector = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

mouse = Controller()
screen_w, screen_h = pyautogui.size()
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

clicking = False
right_clicking = False
last_click_time = 0
prev_x, prev_y = 0, 0
prev_scroll_y = 0

print("🖐️ Air Mouse Started! Press 'q' to quit.")
print(f"🖥️ Screen: {screen_w}x{screen_h}")

# ==========================
# MAIN LOOP
# ==========================
while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands_detector.process(img_rgb)

    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0]
        landmarks = [[lm.x * w, lm.y * h] for lm in hand_landmarks.landmark]
        mp_drawing.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        # Finger tips & key points
        thumb_x, thumb_y = landmarks[4]
        index_x, index_y = landmarks[8]
        middle_x, middle_y = landmarks[12]

        # Map to screen
        screen_x = np.interp(index_x, [50, w - 50], [0, screen_w])
        screen_y = np.interp(index_y, [50, h - 50], [0, screen_h])
        screen_x = int(prev_x * SMOOTHING + screen_x * (1 - SMOOTHING))
        screen_y = int(prev_y * SMOOTHING + screen_y * (1 - SMOOTHING))
        prev_x, prev_y = screen_x, screen_y

        # Move cursor
        mouse.position = (screen_x, screen_y)

        # Finger up/down detection
        finger_tips = [landmarks[4], landmarks[8], landmarks[12], landmarks[16], landmarks[20]]
        finger_pips = [landmarks[3], landmarks[6], landmarks[10], landmarks[14], landmarks[18]]
        fingers_up = [1 if finger_tips[i][1] < finger_pips[i][1] else 0 for i in range(5)]

        # Distance calculations
        thumb_index_dist = math.hypot(index_x - thumb_x, index_y - thumb_y)
        thumb_middle_dist = math.hypot(middle_x - thumb_x, middle_y - thumb_y)

        # ==========================
        # GESTURES
        # ==========================

        # Left Click (Thumb + Index)
        current_time = time.time()
        if thumb_index_dist < PINCH_THRESHOLD and fingers_up[1] == 1:
            if not clicking and (current_time - last_click_time) > CLICK_COOLDOWN:
                mouse.click(Button.left, 1)
                clicking = True
                last_click_time = current_time
                print("🖱️ Left Click")
                cv2.circle(img, (int(index_x), int(index_y)), 15, (0, 255, 0), cv2.FILLED)
        else:
            clicking = False

        # Right Click (Thumb + Middle)
        if thumb_middle_dist < PINCH_THRESHOLD and fingers_up[2] == 1:
            if not right_clicking and (current_time - last_click_time) > CLICK_COOLDOWN:
                mouse.click(Button.right, 1)
                right_clicking = True
                last_click_time = current_time
                print("🖱️ Right Click")
                cv2.circle(img, (int(middle_x), int(middle_y)), 15, (0, 255, 255), cv2.FILLED)
        else:
            right_clicking = False

        # Scroll (Index + Middle Up)
        if fingers_up[1] == 1 and fingers_up[2] == 1 and fingers_up[3] == 0:
            delta_y = index_y - prev_scroll_y
            if abs(delta_y) > 10:  # ignore micro-movement
                direction = -1 if delta_y < 0 else 1
                pyautogui.scroll(direction * SCROLL_SENSITIVITY)
            prev_scroll_y = index_y

        # Debug overlay
        cv2.putText(img, f"Fingers: {fingers_up}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Index Pos: ({int(screen_x)}, {int(screen_y)})", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, f"Thumb-Index Dist: {int(thumb_index_dist)}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 255, 200), 2)

    cv2.imshow("🖐️ Advanced Air Mouse", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
