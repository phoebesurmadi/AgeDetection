import cv2
import numpy as np
from pathlib import Path

def load_dataset(data_dir):
    data_dir = Path(data_dir)
    images = []
    ages = []

    for img_path in data_dir.glob('*.jpg'):
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))  # Resize to a standard size

        # Extract age from filename (assuming format: age_gender_id.jpg)
        age = int(img_path.stem.split('_')[0])

        images.append(img)
        ages.append(age)

    return np.array(images), np.array(ages)

def preprocess_images(images):
    return images.astype(np.float32) / 255.0

if __name__ == "__main__":
    data_dir = "../data/UTKFace"  # Update this path
    images, ages = load_dataset(data_dir)
    processed_images = preprocess_images(images)
    print(f"Loaded {len(images)} images")
    print(f"Image shape: {images[0].shape}")
    print(f"Age range: {ages.min()} - {ages.max()}")