import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
import pickle
import math

# Load the model and class names
MODEL_PATH = "best_model.h5"  # Change to "final_model.h5" if desired
CLASS_NAMES_PATH = "class_names.txt"
IMG_SIZE = 128
GRAYSCALE = True

# Load trained CNN model
model = load_model(MODEL_PATH)

# Load class names
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Load histogram for skin segmentation
with open("hist", "rb") as f:
    hist = pickle.load(f)

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

def preprocess_for_model(img):
    """Resize, grayscale, normalize input image similar to training."""
    if GRAYSCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.astype("float32") / 255.0
    if GRAYSCALE:
        img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img

def merge_bounding_boxes(boxes, threshold=200):
    """Merge bounding boxes if both hands are close."""
    if len(boxes) >= 2:
        (x1_min, y1_min, x1_max, y1_max), (x2_min, y2_min, x2_max, y2_max) = boxes[:2]
        dist = math.hypot((x2_min + x2_max)//2 - (x1_min + x1_max)//2,
                          (y2_min + y2_max)//2 - (y1_min + y1_max)//2)
        if dist < threshold:
            return [(min(x1_min, x2_min), min(y1_min, y2_min),
                     max(x1_max, x2_max), max(y1_max, y2_max))]
    return boxes

def live_predict():
    cap = cv2.VideoCapture(0)
    print("🔍 Live Prediction Started... Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        hand_boxes = []
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                x_list, y_list = [], []
                for lm in handLms.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    x_list.append(x)
                    y_list.append(y)
                x1, y1, x2, y2 = min(x_list), min(y_list), max(x_list), max(y_list)
                hand_boxes.append((x1, y1, x2, y2))
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

        hand_boxes = merge_bounding_boxes(hand_boxes, threshold=200)

        if hand_boxes:
            for (x1, y1, x2, y2) in hand_boxes:
                cv2.rectangle(frame, (x1-10, y1-10), (x2+10, y2+10), (0, 255, 0), 2)
                hand_img = frame[max(0, y1-10):y2+10, max(0, x1-10):x2+10]

                if hand_img.size != 0:
                    input_img = preprocess_for_model(hand_img)
                    prediction = model.predict(input_img)
                    predicted_class = class_names[np.argmax(prediction)]
                    confidence = np.max(prediction)

                    cv2.putText(frame, f"{predicted_class} ({confidence*100:.1f}%)", 
                                (x1, y1-15), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        cv2.imshow("ISL Live Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("🛑 Live Prediction Stopped.")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_predict()
