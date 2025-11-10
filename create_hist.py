import cv2
import numpy as np
import pickle

def create_hist():
    cap = cv2.VideoCapture(0)
    x, y, w, h = 300, 100, 300, 300

    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)

        # ROI to grab skin color data
        roi = frame[y:y+h, x:x+w]
        cv2.rectangle(frame, (x,y), (x+w, y+h), (0,255,0), 2)
        cv2.putText(frame, "Place hand inside the box & press 's' to save histogram", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,255), 2)
        cv2.imshow("Create Histogram", frame)

        key = cv2.waitKey(1)

        if key == ord('s'):
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            # Create the histogram
            hist = cv2.calcHist([roi_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            hist = cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
            with open("hist", "wb") as f:
                pickle.dump(hist, f)
            print("Histogram saved as 'hist'")
            break

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

create_hist()
