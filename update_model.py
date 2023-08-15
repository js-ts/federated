import argparse
import tensorflow as tf
import os
import pandas as pd
import numpy as np
from PIL import Image
import io
from sklearn.model_selection import train_test_split

def load_and_preprocess_data(dataset_path, target_size=(64, 64)):
    data_csv = pd.read_csv(dataset_path)
    
    def bytes_to_image(byte_str):
        image = Image.open(io.BytesIO(byte_str))
        return np.array(image)

    images = [bytes_to_image(eval(row['image'])['bytes']) for _, row in data_csv.iterrows()]
    labels = data_csv['label'].values

    def resize_and_gray_image(image):
        image = Image.fromarray((image * 255).astype(np.uint8))
        image = image.convert('L')  
        return np.array(image.resize(target_size))

    images_processed = [resize_and_gray_image(img) for img in images]
    images_processed = np.array(images_processed).astype('float32') / 255.0
    images_processed = images_processed.reshape(images_processed.shape[0], 64, 64, 1)

    return images_processed, labels

def main(model_path, saved_gradients_dir, dataset_path, save_path):
    # Load the data
    X, y = load_and_preprocess_data(dataset_path)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Load the model
    model = tf.keras.models.load_model(model_path)
    optimizer = tf.keras.optimizers.Adam()

    # Load gradients from saved files
    gradient_files = sorted(os.listdir(saved_gradients_dir))
    loaded_gradients = [np.load(os.path.join(saved_gradients_dir, file)) for file in gradient_files if file.startswith("gradient_")]

    # Convert gradients to tensors
    loaded_gradients = [tf.convert_to_tensor(grad) for grad in loaded_gradients]

    # Check compatibility of shapes and apply gradients
    compatible_shapes = all([tf_var.shape == grad.shape for tf_var, grad in zip(model.trainable_variables, loaded_gradients)])

    if compatible_shapes:
        optimizer.apply_gradients(zip(loaded_gradients, model.trainable_variables))
        print("Gradients applied successfully!")
    else:
        print("Mismatch in shapes detected! Gradients were not applied.")

    # Evaluate the model
    val_loss, val_accuracy = model.evaluate(X_val, y_val)
    print(f"Validation Accuracy: {val_accuracy * 100:.2f}%")

    # Save the model
    model.save(save_path)
    print(f"Model saved to {save_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load a model, apply gradients from saved files, evaluate and save the model.')
    parser.add_argument('--model_path', type=str, default='/brain_tumor_classifier.h5', help='Path to the model file.')
    parser.add_argument('--saved_gradients', type=str, default='/saved_gradients', help='Directory where gradient files are saved.')
    parser.add_argument('--dataset_path', type=str, default='yes-no-brain-tumor-train.csv', help='Path to the dataset.')
    parser.add_argument('--save_path', type=str, default='/outputs/brain_tumor_classifier_updated.h5', help='Path to save the updated model.')

    args = parser.parse_args()
    main(args.model_path, args.saved_gradients, args.dataset_path, args.save_path)
