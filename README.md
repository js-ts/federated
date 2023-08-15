# federated learning docker

## Building the container
```
docker build -t <dockerhub-username>/federated .
```
## Running the container
```
docker run -it --gpus all <dockerhub-username>/federated /bin/bash
```

## Testing the scripts
### Train
```
python train.py --dataset_path brain-tumor-train.csv --model_save_path model.h5
```

### Generate Gradients
```
python gen_gradients.py  --image_dir  /brain_tumor_classifier_dataset --gradients_save_path /gradients --model_path model.h5
```
### Update Model
```
python update_model.py --model_path model.h5 --saved_gradients gradients --dataset_path brain-tumor-train.csv
```

### Inference
```
python inference.py --model_path model.h5 --image_dir brain_tumor_classifier_dataset --output_json outputs.json
```

### Pushing the container
```
docker push <dockerhub-username>/federated
```
