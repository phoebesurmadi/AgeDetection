import cv2
import numpy as np
import tensorflow as tf
from data_loader import preprocess_images

def predict_age(image_path, model_path):
    # Load model
    model = tf.keras.models.load_model(model_path)

    # Load and preprocess image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (224, 224))
    img = preprocess_images(np.array([img]))

    # Predict age
    predicted_age = model.predict(img)[0][0]
    return int(round(predicted_age))

if __name__ == "__main__":
    image_path = "../data/test_image.jpg"  # Update this path
    model_path = "../models/age_model.h5"
    predicted_age = predict_age(image_path, model_path)
    print(f"Predicted age: {predicted_age}")