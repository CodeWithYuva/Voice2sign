"""
Data collection script for word-level sign recognition.

For each word/sign, records multiple video sequences from the webcam,
extracts MediaPipe Holistic keypoints (pose + both hands + face) per frame,
and saves each sequence as a .npy file.

Directory layout produced:
    DATA_DIR/
        hello/
            0.npy   # shape: (SEQUENCE_LENGTH, NUM_FEATURES)
            1.npy
            ...
        thankyou/
            0.npy
            ...

Controls while running:
    's' - start recording a sequence for the current word
    'n' - move to next word
    'q' - quit
"""

import os
import cv2
import numpy as np
import mediapipe as mp

DATA_DIR = "sign_data"          # where sequences are saved
SEQUENCE_LENGTH = 30            # number of frames per sign sequence
SEQUENCES_PER_WORD = 40         # how many recorded samples per word (aim for 40-60+)

# List the word signs you want to collect. Extend this list as needed.
WORDS = ["hello", "thankyou", "please", "yes", "no"]

mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils


def extract_keypoints(results):
    """Flatten pose, face, left-hand, right-hand landmarks into one vector.
    Missing landmarks (e.g. hand not visible) are filled with zeros so the
    output vector always has a fixed length."""
    pose = np.array([[r.x, r.y, r.z, r.visibility] for r in results.pose_landmarks.landmark]).flatten() \
        if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[r.x, r.y, r.z] for r in results.face_landmarks.landmark]).flatten() \
        if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[r.x, r.y, r.z] for r in results.left_hand_landmarks.landmark]).flatten() \
        if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[r.x, r.y, r.z] for r in results.right_hand_landmarks.landmark]).flatten() \
        if results.right_hand_landmarks else np.zeros(21 * 3)
    return np.concatenate([pose, face, lh, rh])


def main():
    cap = cv2.VideoCapture(0)

    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        for word in WORDS:
            word_dir = os.path.join(DATA_DIR, word)
            os.makedirs(word_dir, exist_ok=True)
            existing = len([f for f in os.listdir(word_dir) if f.endswith(".npy")])

            print(f"\n=== Word: '{word}' ({existing} sequences already saved) ===")
            print("Press 's' to record a sequence, 'n' for next word, 'q' to quit.")

            seq_idx = existing
            while seq_idx < SEQUENCES_PER_WORD:
                ret, frame = cap.read()
                if not ret:
                    break

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = holistic.process(image)
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

                cv2.putText(image, f"Word: {word}  Seq: {seq_idx}/{SEQUENCES_PER_WORD}",
                            (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                cv2.putText(image, "Press 's' to record, 'n' next word, 'q' quit",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.imshow("Data Collection", image)

                key = cv2.waitKey(10) & 0xFF
                if key == ord('q'):
                    cap.release()
                    cv2.destroyAllWindows()
                    return
                elif key == ord('n'):
                    break
                elif key == ord('s'):
                    sequence = []
                    for frame_num in range(SEQUENCE_LENGTH):
                        ret, frame = cap.read()
                        if not ret:
                            break
                        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        results = holistic.process(image)
                        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                        cv2.putText(image, f"RECORDING {frame_num + 1}/{SEQUENCE_LENGTH}",
                                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                        cv2.imshow("Data Collection", image)
                        cv2.waitKey(1)

                        sequence.append(extract_keypoints(results))

                    if len(sequence) == SEQUENCE_LENGTH:
                        np.save(os.path.join(word_dir, f"{seq_idx}.npy"), np.array(sequence))
                        print(f"Saved sequence {seq_idx} for '{word}'")
                        seq_idx += 1

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    print(os.getcwd())
    main()