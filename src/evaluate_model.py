import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import numpy as np
from data_loader import load_dataset, preprocess_images

# 1. Plot training history
def plot_history(history):
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='Training MAE')
    plt.plot(history.history['val_mae'], label='Validation MAE')
    plt.title('Model MAE')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()

    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.show()

# 2. Make predictions
def predict_ages(model_path, data_dir):
    # Load the trained model
    model = load_model(model_path)

    # Load and preprocess some test images
    images, true_ages = load_dataset(data_dir)
    images = preprocess_images(np.array(images))

    # Make predictions
    predicted_ages = model.predict(images).flatten()

    # Print results
    for true_age, predicted_age in zip(true_ages[:10], predicted_ages[:10]):
        print(f"True Age: {true_age}, Predicted Age: {predicted_age:.2f}")

    # Calculate and print Mean Absolute Error
    mae = np.mean(np.abs(np.array(true_ages) - predicted_ages))
    print(f"Mean Absolute Error: {mae:.2f} years")

if __name__ == "__main__":
    # Assuming you have the history object saved or returned from training
    # If not, you can skip this part
    # plot_history(history)

    # Make predictions
    model_path = 'best_model.keras'  # or 'final_model.keras', whichever you prefer
    data_dir = "../AgeDetection/data/UTKFace_subset"  # Use a separate test set if available
    predict_ages(model_path, data_dir)