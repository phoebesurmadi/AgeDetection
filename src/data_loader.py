import cv2  # OpenCV for image processing
import numpy as np  # NumPy for numerical operations
from pathlib import Path  # Path for file path operations
import tensorflow as tf  # TensorFlow for data augmentation

# Function to load the dataset
def load_dataset(data_dir):
    data_dir = Path(data_dir)  # Convert data_dir to a Path object
    images = []  # store images
    ages = []  # store ages

    print(f"Searching for images in: {data_dir}")
    for img_path in data_dir.glob('*.jpg'):  # Iterate over all .jpg files in the directory
        print(f"Found image: {img_path}")
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue  # Skip this image if can't load
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert image from BGR to RGB
        img = cv2.resize(img, (224, 224))  # Resize image to 224x224 pixels

        # Extract age from filename (assuming format: age_gender_id.jpg)
        try:
            age = int(img_path.stem.split('_')[0])  # Get age from filename
        except ValueError:
            print(f"Failed to extract age from filename: {img_path}")
            continue  # Skip this image if age couldn't be extracted

        images.append(img)
        ages.append(age)

    print(f"Total images successfully loaded: {len(images)}")
    return np.array(images), np.array(ages)  # Return images and ages as NumPy arrays

# Function to preprocess images
def preprocess_images(images):
    return images.astype(np.float32) / 255.0  # Convert to float32 and normalize to [0, 1] range

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

# Main execution block
if __name__ == "__main__":
    data_dir = "../AgeDetection/data/UTKFace_subset"  # Path to the dataset
    print(f"Looking for images in: {Path(data_dir).resolve()}")  # Print full resolved path

    images, ages = load_dataset(data_dir)

    if len(images) > 0:  # If images were loaded successfully
        processed_images = preprocess_images(images)  # Preprocess the images
        print(f"Loaded {len(images)} images")
        print(f"Image shape: {images[0].shape}")  # Print shape of the first image
        print(f"Age range: {ages.min()} - {ages.max()}")  # Print age range in the dataset
    else:
        print("No images were loaded. Please check the data directory and image files.")