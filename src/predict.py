import numpy as np
from tensorflow.keras.models import load_model
from data_loader import load_dataset

def predict_ages(model_path, data_dir):
    model = load_model(model_path)
    images, true_ages = load_dataset(data_dir)

    predicted_ages = model.predict(images).flatten() * 110  # Denormalize
    true_ages = true_ages * 110  # Denormalize

    for true_age, predicted_age in zip(true_ages[:10], predicted_ages[:10]):
        print(f"True Age: {true_age:.0f}, Predicted Age: {predicted_age:.2f}")

    mae = np.mean(np.abs(true_ages - predicted_ages))
    print(f"Mean Absolute Error: {mae:.2f} years")

if __name__ == "__main__":
    model_path = 'best_model.keras'  # Update this path if necessary
    data_dir = "../AgeDetection/data/UTKFace_subset"  # Update this path if necessary
    predict_ages(model_path, data_dir)