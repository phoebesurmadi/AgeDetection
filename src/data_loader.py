import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations
from pathlib import Path  # Path for file path operations
import tensorflow as tf  # TensorFlow for data augmentation

def load_dataset(data_dir):
    data_dir = Path(data_dir)
    images = []
    ages = []

    print(f"Searching for images in: {data_dir}")

    if not data_dir.exists():
        print(f"Error: Directory {data_dir} does not exist.")
        return images, ages

    image_files = list(data_dir.glob('*.jpg'))
    print(f"Found {len(image_files)} .jpg files in the directory.")

    for img_path in image_files:
        print(f"Processing image: {img_path}")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))  # Resize to a standard size

        # Extract age from filename (assuming format: age_gender_id.jpg)
        try:
            age = int(img_path.stem.split('_')[0])
        except ValueError:
            print(f"Failed to extract age from filename: {img_path}")
            continue

        images.append(img)
        ages.append(age)

    print(f"Total images successfully loaded: {len(images)}")

    # Preprocess the data
    images, ages = preprocess_data(np.array(images), np.array(ages))

    return images, ages

# Function to preprocess images
def preprocess_data(images, ages):
    images = images.astype(np.float32) / 255.0 #Normalize images
    ages = np.array(ages, dtype = np.float32)
    max_age = 110
    ages = ages / max_age # normalize ages to [0,1] range
    return images, ages

# Function to augment images
def augment_image(image):
    # Convert to float32
    image = tf.cast(image, tf.float32)

    # Random flip left-right
    image = tf.image.random_flip_left_right(image)

    # Random brightness adjustment
    image = tf.image.random_brightness(image, max_delta=0.2)

    # Random contrast adjustment
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    # Ensure the image values are still in [0, 1] range
    image = tf.clip_by_value(image, 0.0, 1.0)

    return image

if __name__ == "__main__":
    data_dir = "../AgeDetection/data/UTKFace_subset"  # Update this path if necessary
    print(f"Looking for images in: {Path(data_dir).resolve()}")

    images, ages = load_dataset(data_dir)

    if len(images) > 0:
        print(f"Loaded {len(images)} images")
        print(f"Image shape: {images[0].shape}")
        print(f"Age range: {min(ages)*110:.2f} - {max(ages)*110:.2f}")  # Denormalize ages for display
    else:
        print("No images were loaded. Please check the data directory and image files.")