import os
import cv2
import numpy as np
import math
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import shutil

# Directory paths
DATASET_DIR = "dataset"
PROCESSED_DIR = "processed_dataset"

# Image settings
IMG_SIZE = 128  # Resize to 128x128
GRAYSCALE = True  # Set to False to use RGB

# Create processed dataset directory
if os.path.exists(PROCESSED_DIR):
    shutil.rmtree(PROCESSED_DIR)
os.makedirs(PROCESSED_DIR)

# CLAHE for contrast normalization
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

# Initialize data augmentation generator
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    brightness_range=[0.8, 1.2],
    zoom_range=0.2,
    shear_range=0.1,
    horizontal_flip=False,
    fill_mode='nearest'
)

def add_gaussian_noise(img, std=0.01):
    noise = np.random.normal(0, std, img.shape)
    img_noisy = img + noise
    img_noisy = np.clip(img_noisy, 0, 1)
    return img_noisy

def add_motion_blur(img, kernel_size=15):
    kernel = np.zeros((kernel_size, kernel_size))
    kernel[math.floor(kernel_size/2), :] = 1
    kernel = kernel / kernel_size
    img_blurred = cv2.filter2D(img, -1, kernel)
    return img_blurred

def preprocess_and_augment_image(img_path, label):
    img = cv2.imread(img_path)
    
    if GRAYSCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = clahe.apply(img.astype(np.uint8))
        img = img.astype("float32") / 255.0
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = clahe.apply(img[:,:,0])  # Apply CLAHE on grayscale version
        img = img.astype("float32") / 255.0
    
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    
    # Expand dimension for augmentation (height, width, channels)
    if GRAYSCALE:
        img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    
    # Perform augmentation
    aug_iter = datagen.flow(img, batch_size=1)
    aug_images = [next(aug_iter)[0] for _ in range(5)]  # Generate 5 augmented images (6x total)
    
    # Add Gaussian noise and motion blur to some augmented images
    processed = [img.squeeze()]
    for aug_img in [aug_img.squeeze() for aug_img in aug_images]:
        # Randomly apply noise or blur
        if np.random.random() > 0.5:
            aug_img = add_gaussian_noise(aug_img, std=0.01)
        if np.random.random() > 0.5:
            aug_img = add_motion_blur(aug_img, kernel_size=15)
        processed.append(aug_img)
    
    return processed[:6]  # Return exactly 6 images (1 original + 5 augmented)

def process_dataset():
    for label in os.listdir(DATASET_DIR):
        label_dir = os.path.join(DATASET_DIR, label)
        save_dir = os.path.join(PROCESSED_DIR, label)
        os.makedirs(save_dir, exist_ok=True)

        print(f"Processing label: {label}")
        for img_name in os.listdir(label_dir):
            img_path = os.path.join(label_dir, img_name)

            try:
                processed_images = preprocess_and_augment_image(img_path, label)
                for i, proc_img in enumerate(processed_images):
                    save_path = os.path.join(save_dir, f"{os.path.splitext(img_name)[0]}_aug{i}.npy")
                    np.save(save_path, proc_img)
            
            except Exception as e:
                print(f"Error processing image {img_path}: {e}")

if __name__ == "__main__":
    print("Starting preprocessing of dataset...")
    process_dataset()
    print("Preprocessing complete. Preprocessed dataset saved in:", PROCESSED_DIR)
