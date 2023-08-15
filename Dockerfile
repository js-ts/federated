# Use the official TensorFlow Docker image as the base image
FROM tensorflow/tensorflow:latest

WORKDIR /

# Copy all files from the current directory to the Docker image
COPY . .

# Install any additional dependencies (you can add more if needed)
RUN pip install --no-cache-dir Pillow argparse pandas numpy scikit-learn