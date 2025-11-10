import cv2
import mediapipe as mp
import time
import math


class HandTrackingDynamic:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon
        self.handsMp = mp.solutions.hands
        self.hands = self.handsMp.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findFingers(self, frame, draw=True):
        imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(frame, handLms, self.handsMp.HAND_CONNECTIONS)
        return frame

    def findPosition(self, frame, draw=True):
        handsLmsList = []
        bboxes = []
        if self.results.multi_hand_landmarks:
            for myHand in self.results.multi_hand_landmarks:
                xList, yList = [], []
                lmsList = []
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = frame.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    xList.append(cx)
                    yList.append(cy)
                    lmsList.append([id, cx, cy])
                    if draw:
                        cv2.circle(frame, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                xmin, xmax = min(xList), max(xList)
                ymin, ymax = min(yList), max(yList)
                bbox = (xmin, ymin, xmax, ymax)
                bboxes.append(bbox)
                handsLmsList.append(lmsList)

                if draw:
                    cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (0, 255, 0), 2)
        return handsLmsList, bboxes

    def findCombinedBoundaries(self, frame, bboxes, threshold=100, draw=True):
        if len(bboxes) == 2:
            # Centers of bounding boxes
            bx1, by1 = (bboxes[0][0] + bboxes[0][2]) // 2, (bboxes[0][1] + bboxes[0][3]) // 2
            bx2, by2 = (bboxes[1][0] + bboxes[1][2]) // 2, (bboxes[1][1] + bboxes[1][3]) // 2
            dist = math.hypot(bx2 - bx1, by2 - by1)

            if dist < threshold:
                # Merge both bounding boxes
                xmin = min(bboxes[0][0], bboxes[1][0])
                ymin = min(bboxes[0][1], bboxes[1][1])
                xmax = max(bboxes[0][2], bboxes[1][2])
                ymax = max(bboxes[0][3], bboxes[1][3])

                if draw:
                    cv2.rectangle(frame, (xmin - 20, ymin - 20), (xmax + 20, ymax + 20), (255, 0, 0), 2)

                return [(xmin, ymin, xmax, ymax)]
        return bboxes  # Return original if not merged


def main():
    ctime = 0
    ptime = 0
    cap = cv2.VideoCapture(0)
    detector = HandTrackingDynamic()
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    while True:
        ret, frame = cap.read()
        frame = detector.findFingers(frame)
        handsLmsList, bboxes = detector.findPosition(frame)

        if bboxes:
            combinedBoxes = detector.findCombinedBoundaries(frame, bboxes, threshold=120)
            # You may print boxes for debugging
            # print("Combined Boxes:", combinedBoxes)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime

        cv2.putText(frame, str(int(fps)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
        cv2.imshow('Hand Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
