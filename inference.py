import argparse
import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import json

def load_images(image_dir, target_size=(64, 64)):
    images_list = []
    filenames = []
    for img_name in os.listdir(image_dir):
        img_path = os.path.join(image_dir, img_name)
        if os.path.isfile(img_path):
            image = load_img(img_path, target_size=target_size, color_mode='grayscale')
            image_arr = img_to_array(image) / 255.0
            images_list.append(image_arr)
            filenames.append(img_name)
    return np.array(images_list), filenames

def main(model_path, image_dir, output_json_path):
    # Load the model
    model = tf.keras.models.load_model(model_path)
    
    # Load images
    images, filenames = load_images(image_dir)
    images = images.reshape(images.shape[0], 64, 64, 1)
    
    # Make predictions
    predictions = model.predict(images)
    
    # Assuming binary classification; converting the sigmoid output to binary labels
    binary_predictions = [1 if pred[0] > 0.5 else 0 for pred in predictions]
    
    # Create a dictionary with filenames as keys and predictions as values
    output_dict = dict(zip(filenames, binary_predictions))
    
    # Save predictions to a JSON file
    with open(output_json_path, 'w') as outfile:
        json.dump(output_dict, outfile)

    print(f"Predictions saved to: {output_json_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Perform inference and save outputs to a JSON file.')
    parser.add_argument('--model_path', type=str, required=True, help='Path to the model file.')
    parser.add_argument('--image_dir', type=str, required=True, help='Directory containing images for inference.')
    parser.add_argument('--output_json', type=str, default='predictions.json', help='Path to save predictions in JSON format.')
    
    args = parser.parse_args()
    main(args.model_path, args.image_dir, args.output_json)
