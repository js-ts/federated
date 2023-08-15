import argparse
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import InputLayer
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
import pandas as pd
import numpy as np
from PIL import Image
import io
from sklearn.model_selection import train_test_split

def main(dataset_path, model_save_path):
    # Load the CSV dataset
    data_csv = pd.read_csv(dataset_path)

    # Function to convert byte strings back to numpy arrays
    def bytes_to_image(byte_str):
        image = Image.open(io.BytesIO(byte_str))
        return np.array(image)

    # Extracting byte strings and converting to images
    images = [bytes_to_image(eval(row['image'])['bytes']) for _, row in data_csv.iterrows()]
    labels = data_csv['label'].values

    # Function to resize images and convert them to grayscale
    def resize_and_gray_image(image, target_size=(64, 64)):
        """Resizes the input image to the target size and converts to grayscale."""
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = image.convert('L')  # Convert to grayscale
        return np.array(image.resize(target_size))

    # Resize and convert images to grayscale
    images_processed = [resize_and_gray_image(img) for img in images]

    # Convert list to numpy arrays for compatibility with TensorFlow/Keras
    images_processed = np.array(images_processed).astype('float32') / 255.0

    # Ensure that grayscale images have a single channel dimension
    images_processed = images_processed.reshape(images_processed.shape[0], 64, 64, 1)

    # Splitting the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(images_processed, labels, test_size=0.2, random_state=42)

    model = Sequential()
    model.add(InputLayer(input_shape=(64, 64, 1)))
    model.add(Conv2D(filters=32, kernel_size=3, activation="relu", padding="same"))
    model.add(MaxPooling2D())
    model.add(Conv2D(filters=64, kernel_size=3, activation="relu", padding="same"))
    model.add(MaxPooling2D())
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.3))
    model.add(Dense(64, activation="relu"))
    model.add(BatchNormalization())
    model.add(Dropout(rate=0.3))
    model.add(Dense(1, activation="sigmoid"))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    history = model.fit(X_train, y_train, epochs=9, batch_size=32, validation_data=(X_val, y_val))

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    # Save the model
    model.save(model_save_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model using provided dataset and save the model.')
    parser.add_argument('--dataset_path', type=str, default='yes-no-brain-tumor-train.csv', help='Path to the dataset.')
    parser.add_argument('--model_save_path', type=str, default='brain_tumor_classifier.h5', help='Path to save the trained model.')
    
    args = parser.parse_args()
    main(args.dataset_path, args.model_save_path)
