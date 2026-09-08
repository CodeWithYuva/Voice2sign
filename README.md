# 🤟 Voice2Sign – Indian Sign Language Communication Tool

### 🌍 Bridging the communication gap between Deaf and Non-Speaking Individuals

Voice2Sign is an AI-powered prototype that converts **text ↔ sign language gestures** using **computer vision** and **deep learning**.  
The goal is to make real-time communication easier for individuals using Indian Sign Language (ISL).

---

## 🚀 Features

### 🖐️ ISL to Text
- Detects **two-handed ISL gestures** using your webcam  
- Processes live video feed via **MediaPipe** hand tracking  
- Predicts the corresponding alphabet or word using a **CNN model** trained on custom ISL gesture datasets

### 🔤 Text to ISL
- Converts typed text (A–Z) into sign visuals  
- Displays pre-stored ISL alphabet images dynamically in a Streamlit UI  

### 🧠 Machine Learning
- Custom CNN trained with real ISL gestures  
- Supports grayscale gesture recognition at 128×128 resolution  
- Augmented dataset for better generalization  

---

## 🧩 Tech Stack

| Category | Tools / Libraries |
|-----------|-------------------|
| **Frontend (UI)** | [Streamlit](https://streamlit.io/) |
| **Computer Vision** | [OpenCV](https://opencv.org/), [MediaPipe](https://developers.google.com/mediapipe) |
| **Deep Learning** | [TensorFlow](https://www.tensorflow.org/), [Keras](https://keras.io/) |
| **Data Augmentation** | ImageDataGenerator |
| **Model Training** | Custom CNN with 3 Conv Layers, BatchNorm, Dropout |
| **Language** | Python 3.12 |

---

## ⚙️ Setup Instructions

### 🧱 Step 1 — Clone the Repository
```bash
git clone https://github.com/CodeWithYuva/Voice2sign.git
cd Voice2sign
```
🧰 Step 2 — Create Virtual Environment
```bash
Copy code
python -m venv env
env\Scripts\activate
```
🧩 Step 3 — Install Dependencies
```bash
Copy code
pip install -r requirements.txt
```
🎯 Step 4 — Run the App
```bash
Copy code
streamlit run app.py
```
🧠 Training Your Own Model
Collect gestures using:

```bash
Copy code
python data_collection.py
```
Preprocess and augment images:

```bash
Copy code
python preprocessing.py
```
Train the CNN:

```bash
Copy code
python train_model.py
```
Run live prediction (works standalone):

```bash
Copy code
python live_prediction.py
```

## 📂 Project Structure

```
Voice2sign/
│
├── app.py                     # Streamlit UI
├── live_prediction.py         # Real-time ISL → Text prediction
├── data_collection.py         # Collect custom gesture dataset
├── preprocessing.py           # Preprocessing & data augmentation
├── train_model.py             # CNN model training
├── best_model.h5              # Saved trained model
├── class_names.txt            # Class label mapping
├── signs/                     # Stored ISL alphabet images (A–Z)
├── dataset/                   # Raw gesture captures
├── processed_dataset/         # Preprocessed .npy images
├── requirements.txt           # Dependencies
└── README.md                  # Documentation
```


💡 Future Improvements
🔁 Add real-time voice ↔ sign translation

🗣️ Integrate speech recognition (Google Speech API)

🧍 Add gesture-to-speech output using text-to-speech


👨‍💻 Author
Yuvaraj
📧 jjacky635@gmail.com


