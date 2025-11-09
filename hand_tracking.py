import cv2
from cvzone.HandTrackingModule import HandDetector

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize the Hand Detector
detector = HandDetector(maxHands=1, detectionCon=0.8)

while True:
    success, img = cap.read()
    if not success:
        break

    # Detect hands
    hands, img = detector.findHands(img)  # with drawing

    # If a hand is detected
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]  # List of 21 landmarks
        bbox = hand["bbox"]      # Bounding box info (x, y, w, h)
        centerPoint = hand["center"]

        # Show palm center
        cv2.circle(img, centerPoint, 10, (255, 0, 255), cv2.FILLED)

    cv2.imshow("Hand Tracking", img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
