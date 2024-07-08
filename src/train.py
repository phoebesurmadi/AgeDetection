import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from data_loader import load_dataset
from model import create_age_model

def create_data_generator():
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        zoom_range=0.2
    )

def train_model(data_dir, epochs=100, batch_size=32):
    images, ages = load_dataset(data_dir)

    X_train, X_val, y_train, y_val = train_test_split(images, ages, test_size=0.2, random_state=42)

    model = create_age_model()
    model.compile(optimizer=Adam(learning_rate=0.001), loss='mse', metrics=['mae'])

    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.00001)
    early_stopping = EarlyStopping(patience=15, restore_best_weights=True)
    model_checkpoint = ModelCheckpoint('best_model.keras', save_best_only=True)

    datagen = create_data_generator()
    history = model.fit(
        datagen.flow(X_train, y_train, batch_size=batch_size),
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=[early_stopping, model_checkpoint, reduce_lr]
    )

    # Evaluate model
    test_loss, test_mae = model.evaluate(X_val, y_val)
    print(f"Test MAE: {test_mae * 110:.2f}")  # Denormalize MAE for interpretability

    # Save final model
    model.save('final_model.keras')

    return history, model

if __name__ == "__main__":
    data_dir = "../AgeDetection/data/UTKFace_subset"  # Update this path if necessary
    history, trained_model = train_model(data_dir)
    print("Training completed.")