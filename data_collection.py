import cv2
import numpy as np
import os
import pickle
import mediapipe as mp
import math


def load_hist():
    """Load previously saved skin histogram."""
    with open("hist", "rb") as f:
        return pickle.load(f)


class HandTrackingDynamic:
    """Hand tracking helper class using Mediapipe."""

    def __init__(self, mode=False, maxHands=2, detectionCon=0.7, trackCon=0.7):
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands(
            static_image_mode=mode,
            max_num_hands=maxHands,
            min_detection_confidence=detectionCon,
            min_tracking_confidence=trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, frame):
        """Detect and draw hands."""
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                self.mpDraw.draw_landmarks(frame, handLms, self.handsMp.HAND_CONNECTIONS)
        return frame

    def getHandBoundingBoxes(self, frame):
        """Return list of bounding boxes for detected hands."""
        bboxes = []
        if self.results.multi_hand_landmarks:
            for myHand in self.results.multi_hand_landmarks:
                xList, yList = [], []
                h, w, _ = frame.shape
                for lm in myHand.landmark:
                    xList.append(int(lm.x * w))
                    yList.append(int(lm.y * h))
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bboxes.append((xmin, ymin, xmax, ymax))
        return bboxes

    def mergeBoundingBoxes(self, bboxes, threshold=200):
        """Merge hand boxes if they are close to each other."""
        if len(bboxes) >= 2:
            (x1_min, y1_min, x1_max, y1_max), (x2_min, y2_min, x2_max, y2_max) = bboxes[:2]
            dist = math.hypot((x2_min + x2_max) // 2 - (x1_min + x1_max) // 2,
                              (y2_min + y2_max) // 2 - (y1_min + y1_max) // 2)
            if dist < threshold:
                return [(min(x1_min, x2_min), min(y1_min, y2_min),
                         max(x1_max, x2_max), max(y1_max, y2_max))]
        return bboxes


def collect_gesture_data(label, save_dir="dataset", num_samples=250):
    """Collect dataset of hand gesture using both hands."""
    cap = cv2.VideoCapture(0)
    hist = load_hist()
    detector = HandTrackingDynamic()
    count = 0

    os.makedirs(f"{save_dir}/{label}", exist_ok=True)

    print(f"\n📸 Collecting samples for: {label} (Two-handed support ON)")
    while count < num_samples:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # Detect hands
        frame = detector.findHands(frame)
        bboxes = detector.getHandBoundingBoxes(frame)

        if bboxes:
            bboxes = detector.mergeBoundingBoxes(bboxes, threshold=200)

            for (x1, y1, x2, y2) in bboxes:
                cv2.rectangle(frame, (x1 - 10, y1 - 10), (x2 + 10, y2 + 10), (0, 255, 0), 2)

                # Crop only hand areas
                hand_region = frame[max(0, y1 - 10):y2 + 10, max(0, x1 - 10):x2 + 10]
                if hand_region.size != 0:
                    cv2.imwrite(f"{save_dir}/{label}/{label}_{count}.jpg", hand_region)
                    count += 1
                    cv2.putText(frame, f"Saved: {count}/{num_samples}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Two-Handed Gesture Capture", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(f"\n✅ Finished saving {count} samples for label '{label}'.")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    label = input("Enter the label (e.g., A, Bye, Namaste): ").upper()
    collect_gesture_data(label=label)
