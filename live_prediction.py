import os
os.environ["KERAS_BACKEND"] = "tensorflow"


import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import mediapipe as mp
import pickle
import math

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "best_model.h5")
CLASS_NAMES_PATH = os.path.join(BASE_DIR, "class_names.txt")
IMG_SIZE = 128
GRAYSCALE = True

# Load trained CNN model
model = load_model(MODEL_PATH)

# Load class names
with open(CLASS_NAMES_PATH, 'r') as f:
    class_names = [line.strip() for line in f.readlines()]

# Load histogram for skin segmentation (optional legacy helper)
hist = None
hist_path = os.path.join(BASE_DIR, "hist")
if os.path.exists(hist_path):
    try:
        with open(hist_path, "rb") as f:
            hist = pickle.load(f)
    except Exception:
        hist = None

# Initialize MediaPipe for hand tracking
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# CLAHE contrast normalization matching the training preprocessing pipeline.
# IMPORTANT: Set USE_CLAHE=True ONLY after retraining the model with the new
# CLAHE-based preprocessing.py. The current best_model.h5 was trained WITHOUT
# CLAHE, so keep this False or live accuracy will drop.
USE_CLAHE = False
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if USE_CLAHE else None

def preprocess_for_model(img):
    """Resize, grayscale, optionally CLAHE, normalize input image similar to training."""
    if GRAYSCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if USE_CLAHE and clahe is not None:
            img = clahe.apply(img.astype(np.uint8))
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

def process_single_frame(frame, show_skeleton=True, threshold=85.0):
    """
    Process a single BGR video frame:
    - Detects hand landmarks via MediaPipe.
    - If show_skeleton is True, renders skeletal connections & landmarks.
    - If show_skeleton is False, renders only the hand boundary box & prediction.
    - Predicts sign using CNN model.
    - Returns annotated frame, prediction metadata, and layout data (landmarks & boxes).
    """
    h, w, _ = frame.shape
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    hand_boxes = []
    hand_detected = False
    predicted_class = None
    confidence = 0.0
    landmarks_data = []

    if results.multi_hand_landmarks:
        hand_detected = True
        for handLms in results.multi_hand_landmarks:
            x_list, y_list = [], []
            lms = []
            for lm in handLms.landmark:
                x, y = int(lm.x * w), int(lm.y * h)
                x_list.append(x)
                y_list.append(y)
                lms.append({"x": x, "y": y})
            landmarks_data.append(lms)
            x1, y1, x2, y2 = min(x_list), min(y_list), max(x_list), max(y_list)
            hand_boxes.append((x1, y1, x2, y2))
            
            # Skeleton Diagram ON: Draw MediaPipe hand landmarks and skeletal connections
            if show_skeleton:
                mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    hand_boxes = merge_bounding_boxes(hand_boxes, threshold=200)
    
    boxes_data = []

    if hand_boxes:
        for (x1, y1, x2, y2) in hand_boxes:
            box_x1 = max(0, x1 - 10)
            box_y1 = max(0, y1 - 10)
            box_x2 = min(w, x2 + 10)
            box_y2 = min(h, y2 + 10)

            # Draw hand boundary rectangle
            cv2.rectangle(frame, (box_x1, box_y1), (box_x2, box_y2), (0, 255, 0), 2)
            
            curr_pred = ""
            curr_conf = 0.0
            color = (0, 165, 255)

            hand_img = frame[box_y1:box_y2, box_x1:box_x2]
            if hand_img.size != 0:
                try:
                    input_img = preprocess_for_model(hand_img)
                    prediction = model.predict(input_img, verbose=0)
                    class_idx = int(np.argmax(prediction))
                    curr_conf = float(np.max(prediction) * 100)
                    curr_pred = class_names[class_idx]
                    
                    if curr_conf > confidence:
                        confidence = curr_conf
                        predicted_class = curr_pred

                    # Label color: Green if meets confidence threshold, Orange otherwise
                    color = (0, 255, 0) if curr_conf >= threshold else (0, 165, 255)
                    label_text = f"{curr_pred} ({curr_conf:.1f}%)"
                    
                    # Background pill behind text for readability
                    (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
                    ty = max(25, box_y1 - 10)
                    cv2.rectangle(frame, (box_x1, ty - th - 6), (box_x1 + tw + 8, ty + 4), (0, 0, 0), -1)
                    cv2.putText(frame, label_text, (box_x1 + 4, ty), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                except Exception as e:
                    pass
            
            boxes_data.append({
                "x1": box_x1, "y1": box_y1, "x2": box_x2, "y2": box_y2,
                "label": curr_pred, "confidence": curr_conf, "color": color
            })

    meets_threshold = bool(confidence >= threshold)
    return frame, hand_detected, predicted_class, round(confidence, 1), meets_threshold, {"landmarks": landmarks_data, "boxes": boxes_data}


def live_predict(show_skeleton=True):
    cap = cv2.VideoCapture(0)
    print("🔍 Live Prediction Started... Press 'q' to exit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame, _, _, _, _, _ = process_single_frame(frame, show_skeleton=show_skeleton)
        cv2.imshow("ISL Live Prediction", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("🛑 Live Prediction Stopped.")
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    live_predict()
