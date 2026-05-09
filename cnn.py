import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

PROCESSED_DIR = "C:\\Users\\HP\\Desktop\\ISL prototype\\lil\\processed_dataset"
IMG_SIZE = 128     # Input image size
GRAYSCALE = True   # Set this based on the preprocessing stage
NUM_AUG = 4        # 1 original + 3 augmentations per image

def load_data(processed_dir=PROCESSED_DIR):
    X, y = [], []
    classes = sorted(os.listdir(processed_dir))
    class_to_idx = {cls: i for i, cls in enumerate(classes)}

    for cls in classes:
        cls_dir = os.path.join(processed_dir, cls)
        for npy_file in os.listdir(cls_dir):
            npy_path = os.path.join(cls_dir, npy_file)

            # Skip directories
            if os.path.isdir(npy_path):
                print(f"Skipping directory: {npy_path}")
                continue

            try:
                img_array = np.load(npy_path)
                X.append(img_array)
                y.append(class_to_idx[cls])
            except Exception as e:
                print(f"Error loading {npy_path}: {e}")

    X = np.array(X)
    y = to_categorical(np.array(y), num_classes=len(classes))
    return train_test_split(X, y, test_size=0.2, random_state=42), classes

def build_model(input_shape, num_classes):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(64, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Conv2D(128, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D(pool_size=(2, 2)),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def plot_history(history):
    plt.figure(figsize=(12, 5))

    # Plot train/validation accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc')
    plt.plot(history.history['val_accuracy'], label='Val Acc')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    # Plot train/validation loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Load preprocessed data
    (X_train, X_test, y_train, y_test), class_names = load_data()

    # Reshape if grayscale (add channel dim)
    if GRAYSCALE:
        X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
        X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    else:
        X_train = X_train.reshape(-1, IMG_SIZE, IMG_SIZE, 3)
        X_test = X_test.reshape(-1, IMG_SIZE, IMG_SIZE, 3)

    # Build model
    input_shape = X_train.shape[1:]
    model = build_model(input_shape, len(class_names))

    # Callbacks
    checkpoint = ModelCheckpoint("best_model.h5", monitor='val_accuracy', save_best_only=True)
    early_stop = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=15,
        batch_size=32,
        callbacks=[checkpoint, early_stop]
    )

    # Plot training history
    plot_history(history)

    # Evaluate and save final model
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
    print(f"Test Accuracy: {test_acc * 100:.2f}%")
    model.save("final_model.h5")

    # Save class names
    with open('class_names.txt', 'w') as f:
        for cls in class_names:
            f.write(cls + '\n')
