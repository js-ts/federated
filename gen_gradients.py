import argparse
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import numpy as np
import uuid

def main(image_dir, model_path, gradients_save_path):
    # Load all images from directory into a list
    target_size = (32, 32)
    images_list = []
    for img_path in os.listdir(image_dir):
        full_path = os.path.join(image_dir, img_path)
        if os.path.isfile(full_path):
            image = load_img(full_path, target_size=(64, 64), color_mode='grayscale')
            image_arr = img_to_array(image) / 255.0
            images_list.append(image_arr)

    data = np.array(images_list)

    # Load the model
    model = tf.keras.models.load_model(model_path)
    model.summary()

    # Check if data is available and is not empty
    if data is not None and len(data) > 0:
        pseudo_labels = model.predict(data)
    else:
        print("The data variable is empty!")

    def compute_gradients(model, data, labels):
        with tf.GradientTape() as tape:
            predictions = model(data, training=True)
            loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
        gradients = tape.gradient(loss, model.trainable_variables)
        return gradients

    gradients = compute_gradients(model, data, pseudo_labels)

    # Serialize gradients and save to files
    os.makedirs(gradients_save_path, exist_ok=True)

    for grad in gradients:
        gradient_id = uuid.uuid4()
        path = os.path.join(gradients_save_path, f'gradient_{gradient_id}.npy')
        np.save(path, grad.numpy())

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Load images, use model to predict and compute gradients.')
    parser.add_argument('--image_dir', type=str, default='/content/brain_tumor_dataset', help='Directory where images are located.')
    parser.add_argument('--model_path', type=str, default='/content/brain_tumor_classifier.h5', help='Path to the model file.')
    parser.add_argument('--gradients_save_path', type=str, default='saved_gradients', help='Directory where gradients will be saved.')
    
    args = parser.parse_args()
    main(args.image_dir, args.model_path, args.gradients_save_path)
