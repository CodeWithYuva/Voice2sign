# 📘 Project Documentation — ISL Communication Tool (Voice2Sign Prototype)

**Full technical documentation for the Indian Sign Language (ISL) Communication Tool.**

This document covers the entire project: file structure, what each file does, the end-to-end project flow, every library used, the main methods/classes, the exact CNN model specifications (verified from the saved model weights), and the dataset layout.

---

## 1. Project Overview

The **ISL Communication Tool** is an AI-powered prototype that bridges communication between Deaf/Non-Speaking individuals and the hearing world by converting **text ↔ sign language gestures**.

It has two functional modes:

| Mode | Direction | How it works |
|------|-----------|--------------|
| 🔤 **Text to ISL** | Text → Sign visuals | User types text (A–Z); the app displays the corresponding pre-stored sign images. |
| 🖐️ **ISL to Text** | Live gesture → Text | Uses the webcam with **MediaPipe** hand tracking + a **custom CNN** to recognize ISL gestures in real time. |

The prototype is a local desktop application with a **Streamlit** frontend and an **OpenCV** live-video window.

---

## 2. Tech Stack & Libraries

The dependencies are pinned in [`requirements.txt`](requirements.txt). They are carefully versioned for stability (Python 3.12.3 recommended).

| Category | Library | Version / Pin | Purpose |
|----------|---------|---------------|---------|
| **Deep Learning** | TensorFlow / Keras | `tensorflow==2.16.1` | CNN training, loading, and inference |
| | `protobuf` | `4.25.3` | **Critical bridge** — the exact version compatible with both TF and MediaPipe |
| | NumPy | `<2.0.0` | Array math (TF 2.x breaks with NumPy 2.0+) |
| **Computer Vision** | `mediapipe` | `0.10.13` | Real-time two-handed landmark tracking |
| | `opencv-contrib-python` | latest | Webcam capture, ROI cropping, drawing |
| | Pillow | latest | Image loading in the Streamlit UI |
| **UI** | `streamlit` | latest | Web-based frontend (Text ↔ ISL tabs) |
| **Data** | `scikit-learn` | latest | `train_test_split` |
| | `matplotlib` | `3.9.0` | Training-history plots (pinned for Streamlit stability) |

> **Version note:** the `protobuf` pin is the magic ingredient. TensorFlow 2.16, MediaPipe 0.10.13, and Protobuf 4.25.x are the last mutually-compatible combination.

---

## 3. Project File Structure

```
lil/
├── app.py                    # Streamlit UI (Text→ISL + ISL→Text launcher)
├── live_prediction.py        # Real-time ISL → Text recognition (webcam)
├── data_collection.py        # Webcam dataset collection (two-handed)
├── preprocessing.py          # Resize / grayscale / normalize / augment
├── cnn.py                    # CNN model definition + training (creates best_model.h5, final_model.h5)
├── test.py                   # Standalone MediaPipe hand-tracking dev harness
├── create_hist.py            # Builds skin-color HSV histogram → 'hist'
├── Untitled7.ipynb           # Google Colab training notebook (GPU / T4)
│
├── best_model.h5             # Trained CNN — best epoch (by val_accuracy) — USED in live prediction
├── final_model.h5            # Trained CNN — last-epoch + evaluated model (same architecture)
├── class_names.txt           # 18 class labels in sorted order (used for prediction decoding)
├── hist                      # Skin-segmentation HSV histogram (pickle, 180×256)
│
├── dataset/                  # RAW captured gesture images (18 label folders)
├── processed_dataset/        # Preprocessed + augmented samples as .npy arrays
├── processed_dataset.zip     # Zip of processed_dataset (uploaded to Colab)
│
├── signs/
│   └── signs/                # 35 static sign images (A–Z + digits 1–9) used by Text→ISL
├── signs.zip                 # Zip of signs/ (distribution backup)
│
├── requirements.txt          # Pinned dependencies
├── README.md                 # User-facing readme (overview + setup)
├── .gitignore                # Ignores env/, dataset/, processed_dataset/, zips, __pycache__
├── env/                      # Local Python 3.12 virtual environment (not versioned)
└── PROJECT_DOCUMENTATION.md  # This document
```

> **Note:** The `README.md` mentions a `train_model.py` script — the actual training script in this repository is **`cnn.py`**.

---

## 4. File-by-File Analysis

### 4.1 `data_collection.py` — Gesture Dataset Collection
Collects a custom gesture dataset from the webcam, with **two-handed support**.

- **`HandTrackingDynamic`** class — MediaPipe wrapper:
  - `findHands(frame)` — detects and draws hand landmarks
  - `getHandBoundingBoxes(frame)` — returns `(xmin, ymin, xmax, ymax)` for each hand
  - `mergeBoundingBoxes(bboxes, threshold=200)` — if two hands are closer than **200 px** center-to-center, merges them into one box (needed for two-handed ISL signs)
- **`collect_gesture_data(label, save_dir="dataset", num_samples=100)`** — main capture loop:
  - Captures frames, finds hands, merges nearby hands, draws a padded green box (`x1-10`..`y2+10`)
  - Crops the hand region and saves it as `dataset/<LABEL>/<LABEL>_<count>.jpg`
  - Press **`q`** to stop early

Run: `python data_collection.py` → prompts for a label (e.g., `A`, `Bye`, `Namaste`).

### 4.2 `preprocessing.py` — Preprocessing + Augmentation
Turns raw `.jpg` captures into normalized, augmented `.npy` arrays.

- Settings: `IMG_SIZE = 128`, `GRAYSCALE = True`
- **`ImageDataGenerator`** augmentation config (applied at runtime):
  - `rotation_range=15`, `width/height_shift_range=0.15`, `brightness_range=[0.8, 1.2]`, `zoom_range=0.2`, `horizontal_flip=True`, `fill_mode='nearest'`
- **`preprocess_and_augment_image(img_path, label)`**:
  1. Read with OpenCV → grayscale (if `GRAYSCALE`) → resize to **128×128** → normalize to `[0, 1]` (`/255.0`)
  2. Expand dims and run 3 augmentations
  3. Returns **4 images**: 1 original + 3 augmented
- **`process_dataset()`** — iterates every `dataset/<label>` folder and saves each sample as:
  `<name>_aug0.npy` (original), `_aug1/_aug2/_aug3.npy` (augmented)
- The old `processed_dataset/` is wiped and rebuilt each run.

Run: `python preprocessing.py`

### 4.3 `cnn.py` — CNN Training Script
Defines, trains, and exports the ISL classifier.

- **`load_data(processed_dir)`** — loads all `.npy` arrays, builds one-hot labels via `to_categorical`, returns 80/20 train/test split (`test_size=0.2, random_state=42`) plus sorted class names.
- **`build_model(input_shape, num_classes)`** — the CNN architecture (see Section 6).
- **`plot_history(history)`** — two-panel matplotlib plot (accuracy + loss).
- **Training protocol** (in `__main__`):
  - Reshape to `(128, 128, 1)` grayscale
  - Callbacks: `ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True)` + `EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)`
  - `model.fit(...)`, `validation_split=0.2`, `epochs=15`, `batch_size=32`
  - Evaluate on the test split → saves `final_model.h5` and writes `class_names.txt`

Run: `python cnn.py`

### 4.4 `live_prediction.py` — Real-Time ISL → Text
The standalone recognition engine (imported by `app.py` or run directly).

- **Loads at startup:** `best_model.h5`, `class_names.txt`, and the `hist` pickle.
- **`preprocess_for_model(img)`** — mirrors training preprocessing: grayscale → resize 128×128 → `float32 / 255.0` → add channel dim `(1, 128, 128, 1)`.
- **`merge_bounding_boxes(boxes, threshold=200)`** — same two-hand merge logic as collection.
- **`live_predict()`** — the main loop:
  1. Opens `cv2.VideoCapture(0)`, flips frame
  2. MediaPipe `Hands(max_num_hands=2, min_detection_confidence=0.7)` → landmarks
  3. Computes bounding boxes, draws landmarks, merges adjacent hands
  4. Crops padded hand ROI → `preprocess_for_model` → `model.predict`
  5. Decodes `class_names[np.argmax(pred)]`, draws `"CLASS (conf%)"` label
  6. Press **`q`** to stop

Run standalone: `python live_prediction.py` (no Streamlit needed).

### 4.5 `app.py` — Streamlit Frontend
The UI entry point.

- Imports `live_predict` from `live_prediction` (with a graceful fallback if import fails).
- **Tab 1 — "Text to ISL"**: reads input text (`.upper()`), and for each character displays `signs/signs/<CHAR>.jpg` via Pillow/PIL. Shows a warning for missing/unsupported characters.
- **Tab 2 — "ISL to Text"**: a button that launches `live_predict()` in a separate OpenCV window.

Run: `streamlit run app.py`

### 4.6 `test.py` — Hand-Tracking Dev Harness
A standalone experiment/testing script for MediaPipe hand tracking (not part of the main pipeline).

- `findFingers(frame)` — draws landmarks
- `findPosition(frame)` — returns landmark list + bounding boxes, draws hand boxes
- `findCombinedBoundaries(frame, bboxes, threshold=100, draw=True)` — merges two hand boxes when closer than 100 px (blue merged box)
- Displays live **FPS** overlay. Camera set to 640×480.

### 4.7 `create_hist.py` — Skin Histogram Creation
Generates the `hist` file used for skin segmentation.

- Runs the webcam, shows a green ROI box at `x=300, y=100, w=300, h=300`
- On **`s`**: converts ROI to HSV, computes a 2D histogram on channels H (0–180) and S (0–256): `(180, 256)`, normalizes to 0–255, saves via `pickle` to `hist`
- On **`q`**: quits

### 4.8 `Untitled7.ipynb` — Colab Training Notebook
Google Colab notebook used to train on GPU (T4 accelerator).

- Cell 1: installs TensorFlow/matplotlib
- Cell 2: unzips `processed_dataset.zip`
- Cell 3: contains the **exact same code as `cnn.py`** but with `epochs=30` and Colab paths. Its captured output documents:
  - Early experimental **2-class** run: val accuracy ~81.8%, **test accuracy 83.33%**
  - The full printed `model.summary()` we use in Section 6 (13 layers, 3,305,218 params)

> The `best_model.h5` / `final_model.h5` shipped here are 18-class models (see weights analysis in Section 6), i.e., the final trained artifacts.

### 4.9 Model / Data artifact files

| File | Type | Details |
|------|------|---------|
| `best_model.h5` | Keras HDF5 model | 18-class CNN, best epoch by val_accuracy — **used by live prediction** |
| `final_model.h5` | Keras HDF5 model | Same architecture, last-epoch weights, saved after test evaluation |
| `class_names.txt` | Text (1 label/line) | `1 2 3 4 5 6 8 A B C D E F G I L O V` (18 classes, sorted) |
| `hist` | Pickled numpy array | Skin histogram, shape `(180, 256)`, dtype `float32`, values 0–255 |
| `dataset/` | Raw images | 18 folders, 65–249 images per class |
| `processed_dataset/` | `.npy` arrays | 128×128 grayscale floats `[0,1]`, ~10 384 samples (4× raw) |

> **Note (dead code):** `hist`, `load_hist()` in `data_collection.py`, and the `hist` load in `live_prediction.py` are **loaded but never applied** — the pipeline currently segments hands purely via MediaPipe bounding boxes. `create_hist.py` remains a legacy helper.

---

## 5. Project Flow / Pipeline

```
   [ STAGE 1 ]              [ STAGE 2 ]                [ STAGE 3 ]                 [ STAGE 4 ]
 DATA COLLECTION         PREPROCESSING                 TRAINING                    INFERENCE
+---------------+      +------------------+      +-------------------+       +------------------------+
| data_collection.py    preprocessing.py             cnn.py  / Colab           live_prediction.py
| webcam → MediaPipe    128×128 resize      →       80/20 train/test split      webcam → MediaPipe
| hands → merge boxes    grayscale          →       CNN (18 classes)           hands → merge boxes
| crop ROI →            normalize /255      →       ModelCheckpoint(+best)     crop ROI → preprocess
| dataset/<label>/*.jpg  ImageDataGenerator →       EarlyStopping(p=5)         model.predict
|                        _aug0..3 .npy      →       best_model.h5              CLASS (conf%) overlay
+---------------+      +------------------+          final_model.h5            class_names.txt
                                                    class_names.txt
                                           ↓
+---------------------------------------------------+
|  app.py (Streamlit)                            |
|  Tab 1: Text → ISL  = signs/signs/{char}.jpg |
|  Tab 2: ISL → Text  = calls live_predict()   |
+---------------------------------------------------+
```

**End-to-end summary:**
1. **Collect** gestures with `data_collection.py` → raw `.jpg`s in `dataset/`
2. **Preprocess** with `preprocessing.py` → normalized, augmented `.npy` in `processed_dataset/` (4 samples per raw image)
3. **Train** with `cnn.py` (or the Colab notebook from a zip upload) → produce `best_model.h5`, `final_model.h5`, `class_names.txt`
4. **Use** via `streamlit run app.py` — live recognition runs in an OpenCV window; text conversion displays static sign images.

---

## 6. CNN Model Specifications

Verified directly from the shipped H5 weight files (`best_model.h5` and `final_model.h5` — both identical in structure).

### 6.1 Architecture (13 layers, Sequential)

| # | Layer | Output Shape | Params | Activation |
|---|-------|--------------|--------|------------|
| 1 | Conv2D (32 filters, 3×3) | (126, 126, 32) | 320 | ReLU |
| 2 | BatchNormalization | (126, 126, 32) | 128 | — |
| 3 | MaxPooling2D (2×2) | (63, 63, 32) | 0 | — |
| 4 | Conv2D (64 filters, 3×3) | (61, 61, 64) | 18,496 | ReLU |
| 5 | BatchNormalization | (61, 61, 64) | 256 | — |
| 6 | MaxPooling2D (2×2) | (30, 30, 64) | 0 | — |
| 7 | Conv2D (128 filters, 3×3) | (28, 28, 128) | 73,856 | ReLU |
| 8 | BatchNormalization | (28, 28, 128) | 512 | — |
| 9 | MaxPooling2D (2×2) | (14, 14, 128) | 0 | — |
| 10 | Flatten | 25,088 | 0 | — |
| 11 | Dense (128) | 128 | 3,211,392 | ReLU |
| 12 | Dropout (0.5) | 128 | 0 | — |
| 13 | Dense (18) | 18 | 258 | Softmax |

### 6.2 Parameter summary

| Metric | Value |
|--------|-------|
| Total params | **3,305,218** (≈12.61 MB) |
| Trainable params | 3,304,770 |
| Non-trainable params | 448 (BatchNorm moving stats) |

> Weight shapes confirm: `conv2d/kernel (3, 3, 1, 32)`, `conv2d_1 (3, 3, 32, 64)`, `conv2d_2 (3, 3, 64, 128)`, `dense/kernel (25088, 128)`, `dense_1/kernel (128, 18)` → **input is single-channel grayscale**.

### 6.3 Input / output contract

- **Input:** `(128, 128, 1)` float32 grayscale, normalized to `[0, 1]`
- **Output:** 18 classes (Softmax) = `class_names.txt`
  `['1','2','3','4','5','6','8','A','B','C','D','E','F','G','I','L','O','V']`
- Prediction decoding: `class_names[np.argmax(probabilities)]`, confidence = `np.max(probabilities)`

### 6.4 Training configuration

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | Adam (default, lr≈0.001) |
| Loss | Categorical cross-entropy |
| Metric | Accuracy |
| Data split | 80/20 train/test (`random_state=42`) plus inner `validation_split=0.2` |
| Batch size | 32 |
| Epochs | 15 (`cnn.py`) / 30 (Colab notebook) |
| Early stopping | patience = 5 on `val_loss`, `restore_best_weights=True` |
| Model checkpoint | save best on `val_accuracy` → `best_model.h5` |
| Augmentation | Geometric + photometric via `ImageDataGenerator` (see 4.2) |
| Reported accuracy | Early 2-class Colab run: **83.33%** test accuracy |

### 6.5 Design rationale
- **3 convolutional blocks** of increasing depth (32→64→128) extract low-to-high level hand/shape features.
- **BatchNorm after every conv** stabilizes training.
- **Dropout(0.5)** before the final layer prevents overfitting on the small custom dataset.
- **Grayscale 128×128** input keeps the model small (~3.3M params) for real-time inference on low-end CPUs.

---

## 7. Dataset Details

- **18 classes** (digits `1,2,3,4,5,6,8` + letters `A,B,C,D,E,F,G,I,L,O,V`) — omitted letters are dynamic/two-handed gestures not yet collected.
- **Raw dataset** (`dataset/`): 2,574 `.jpg` images, 65–249 per class:

| Label | Raw | Augmented | Label | Raw | Augmented |
|-------|-----|-----------|-------|-----|-----------|
| 1 | 224 | 900 | B | 249 | 1,000 |
| 2 | 226 | 908 | C | 99 | 400 |
| 3 | 249 | 1,000 | D | 93 | 380 |
| 4 | 99 | 400 | E | 99 | 400 |
| 5 | 99 | 400 | F | 99 | 400 |
| 6 | 98 | 400 | G | 108 | 436 |
| 8 | 99 | 400 | I | 99 | 400 |
| A | 65 | 268 | L | 248 | 1,000 |
| — | — | — | O | 72 | 292 |
| — | — | — | V | 249 | 1,000 |

- **Augmented dataset** (`processed_dataset/`): ~10,384 `.npy` samples (each raw image × 4: `_aug0` original + `_aug1`–`_aug3`).
- Class imbalance exists (e.g., A=65 vs B=249 raw) — a future improvement target.
- `signs/signs/` contains **35 static images** (A–Z + 1–9) purely for the Text→ISL display, independent of the trained 18 classes.

---

## 8. Key Methods, Classes & Techniques

| Concept | Where | Notes |
|---------|-------|-------|
| `HandTrackingDynamic` class | `data_collection.py`, `test.py` | MediaPipe wrapper; reused design (collection implements a newer variant) |
| Two-hand bounding-box merge | `mergeBoundingBoxes` / `merge_bounding_boxes` | Euclidean center distance < **200 px** (prediction) / 100 px (test.py) → merge into one ROI for two-handed signs |
| Data augmentation at runtime | `preprocessing.py` | `ImageDataGenerator.flow()` — outputs to disk as `.npy` |
| One-hot encoding | `cnn.py` `load_data()` | `to_categorical` |
| Model persistence | `cnn.py` | HDF5 `.h5` via `model.save()` (Keras 3 legacy format) |
| Real-time inference | `live_prediction.py` | frame loop: detect → crop → preprocess → predict → overlay |
| Confidence display | `live_prediction.py` | `np.max(prediction)` shown to the user |
| Streamlit tabs | `app.py` | Two independent modes in one UI |
| Skin segmentation histogram | `create_hist.py` → `hist` | Legacy/optional; currently unused in main flow |

---

## 9. How to Run

```powershell
# 1. Create & activate virtual env (Python 3.12.3)
python -m venv env
env\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3a. Run the full app (Streamlit UI)
streamlit run app.py

# 3b. Run live prediction standalone (no UI)
python live_prediction.py

# 4. Retrain from scratch
python data_collection.py    # collect a new class, e.g.  A
python preprocessing.py      # augment everything
python cnn.py                # train  →  best_model.h5 / final_model.h5 / class_names.txt
```

**Colab training path:** zip `processed_dataset/` → `processed_dataset.zip`, upload to `Untitled7.ipynb`, run cells on a T4 GPU, download generated `.h5` + `class_names.txt` back to the repo.

---

## 10. Known Limitations & Improvement Ideas

- ❗ `hist` (skin segmentation) is dead code — pipeline relies on MediaPipe boxes only.
- ⚠️ Class imbalance in the dataset (65–249 images per class).
- ⚠️ Only 18 of 26 letters covered; remaining letters are two-handed/dynamic.
- 💡 Future: speech recognition (voice→sign), text-to-speech (sign→audio), web deployment, teacher-forcing/LSTM for continuous signs.