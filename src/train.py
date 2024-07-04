import tensorflow as tf
from data_loader import load_dataset, preprocess_images
from model import create_age_model
from sklearn.model_selection import train_test_split

def train_model(data_dir, epochs=50, batch_size=32):
    # Load and preprocess data
    images, ages = load_dataset(data_dir)
    images = preprocess_images(images)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(images, ages, test_size=0.2, random_state=42)

    # Create and compile model
    model = create_age_model()
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])

    # Train model
    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs,
        batch_size=batch_size
    )

    # Evaluate model
    test_loss, test_mae = model.evaluate(X_test, y_test)
    print(f"Test MAE: {test_mae}")

    # Save model
    model.save('../models/age_model.h5')

    return history

if __name__ == "__main__":
    data_dir = "../data/UTKFace"  # Update this path
    history = train_model(data_dir)