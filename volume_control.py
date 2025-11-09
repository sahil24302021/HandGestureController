import cv2
from cvzone.HandTrackingModule import HandDetector
import math
import os

# Initialize webcam
cap = cv2.VideoCapture(0)
detector = HandDetector(detectionCon=0.8, maxHands=1)

# Helper function to set system volume (macOS)
def set_volume(level):
    # Volume range: 0 (mute) - 100 (max)
    level = max(0, min(100, level))
    os.system(f"osascript -e 'set volume output volume {int(level)}'")

vol = 50
set_volume(vol)

while True:
    success, img = cap.read()
    if not success:
        break

    hands, img = detector.findHands(img)
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]

        # Thumb tip = 4, Index finger tip = 8
        x1, y1 = lmList[4][:2]
        x2, y2 = lmList[8][:2]

        # Draw line between them
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 3)
        length = math.hypot(x2 - x1, y2 - y1)

        # Convert distance to volume range
        vol = int((length - 30) / (250 - 30) * 100)
        set_volume(vol)
        print(f"Volume: {vol}%")

        cv2.putText(img, f'Volume: {vol}%', (40, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 3)

    cv2.imshow("Hand Volume Control", img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
