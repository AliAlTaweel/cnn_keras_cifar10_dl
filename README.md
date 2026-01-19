# CIFAR-10 CNN Classifier

**Project:** `cnn_keras_cifar10_dl`

## Overview

This repository contains a compact Convolutional Neural Network (CNN) implementation to classify images from the CIFAR-10 dataset. The goal is to provide an easy-to-run project skeleton (data loading, model definition, training loop, evaluation and plotting), suitable as a learning resource or a starting point for experiments.

The project includes both a command-line training interface and a **FastAPI REST API** for serving predictions.

## Contents

- `src/model.py` — model architecture definitions
- `src/app.py` — FastAPI application for serving predictions
- `src/predict.py` — main training and evaluation code (functions: `load_data`, `build_model`, `train`, `evaluate`)
- `tests/test_model.py` — basic tests that ensure model and data shapes are correct
- `Makefile` — convenience commands (`train`, `test`, `clean`, `api`)
- `requirements.txt` — runtime dependencies
- `requirements.txt` — development dependencies

## Dataset

**CIFAR-10** is a standard benchmark dataset of 60,000 32×32 colour images in 10 classes, with 50,000 training images and 10,000 test images.

The script uses a Google Cloud mirror to download CIFAR-10 (faster and more reliable):

```
https://storage.googleapis.com/tensorflow/tf-keras-datasets/cifar-10-batches-py.tar.gz
```

### Classes

The dataset contains 10 classes:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

## Model Architecture

The project provides two model architectures:

### Simple Model (Notebook Architecture)

```
Conv2D(32) -> MaxPool
Conv2D(64) -> MaxPool
Conv2D(64) -> Flatten -> Dense(64) -> Dense(10)
```

### Enhanced Model (Recommended)

```
Conv2D(32) x2 -> BatchNorm -> MaxPool -> Dropout(0.2)
Conv2D(64) x2 -> BatchNorm -> MaxPool -> Dropout(0.3)
Conv2D(128) x2 -> BatchNorm -> MaxPool -> Dropout(0.4)
Flatten -> Dense(128) -> BatchNorm -> Dropout(0.5) -> Dense(10)
```

Both models are compiled with **Adam optimizer** and `SparseCategoricalCrossentropy(from_logits=True)` loss.

## Installation

### Prerequisites

- Python 3.8+
- pip

### Setup

1. Clone the repository:

```bash
git clone https://github.com/AliAlTaweel/cnn_keras_cifar10_dl.git
cd cnn_keras_cifar10_dl
```

2. Install dependencies:

```bash
make install
# or
pip install -r requirements.txt
```

3. Create project structure:

```bash
make setup
# or
python create_structure.py
```

## Training

### Quick Start

Train with default settings (enhanced model):

```bash
make train
# or
python -m source.train --epochs 10 --batch-size 64
```

Train with simple model (matching notebook):

```bash
make train-simple
# or
python -m source.train --epochs 10 --batch-size 64 --simple
```

### Training Options

```bash
python -m source.train [OPTIONS]

Options:
  --epochs INT        Number of training epochs (default: 10)
  --batch-size INT    Batch size for training (default: 64)
  --simple            Use simple model architecture
  --eval-only         Only evaluate existing model
  --checkpoint PATH   Path to model checkpoint for evaluation
```

### Training Features

The training routine includes:

- **Model checkpointing**: Saves best model based on validation accuracy
- **Early stopping**: Stops training if validation loss doesn't improve
- **Learning rate scheduling**: Reduces learning rate on plateau
- **CSV logging**: Saves training history to `logs/training_log.csv`
- **Automatic plotting**: Generates and saves:
  - `accuracy_curve.png` - Training and validation accuracy
  - `loss_curve.png` - Training and validation loss
  - `confusion_matrix.png` - Confusion matrix heatmap
  - `predictions_visualization.png` - Sample predictions

## Evaluation & Metrics

After training, the script computes predictions on the test set and prints:

- **Classification report** (precision, recall, f1-score for each class)
- **Confusion matrix** (saved as a heatmap)
- **Per-class accuracy**

Evaluate an existing model:

```bash
make eval
# or
python -m source.train --eval-only
```

## FastAPI REST API

The project includes a production-ready REST API for serving predictions.

### Start the API Server

```bash
make api
# or
uvicorn source.api.app:app --host 0.0.0.0 --port 8000 --reload
```

### API Endpoints

#### 1. Health Check

```bash
GET /
GET /health
```

#### 2. Model Information

```bash
GET /model/info
```

Response:

```json
{
  "model_name": "CIFAR-10 CNN Classifier",
  "version": "1.0.0",
  "input_shape": [32, 32, 3],
  "num_classes": 10,
  "class_names": ["airplane", "automobile", ...],
  "is_loaded": true
}
```

#### 3. Single Image Prediction

```bash
POST /predict
Content-Type: multipart/form-data
```

Example using curl:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@/path/to/image.jpg"
```

Response:

```json
{
  "predicted_class": "airplane",
  "predicted_class_index": 0,
  "confidence": 0.95,
  "all_probabilities": {
    "airplane": 0.95,
    "automobile": 0.02,
    "bird": 0.01,
    ...
  }
}
```

#### 4. Batch Prediction

```bash
POST /predict/batch
Content-Type: multipart/form-data
```

#### 5. Get Classes

```bash
GET /classes
```

### API Documentation

Interactive API documentation is available at:

- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

### Example Python Client

```python
import requests

# Predict single image
with open('test_image.jpg', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:8000/predict', files=files)
    print(response.json())

# Get model info
response = requests.get('http://localhost:8000/model/info')
print(response.json())
```

## Tests

Run tests with:

```bash
make test
# or
pytest -v
# or
python -m unittest discover -v
```

The test suite includes:

- Model architecture validation
- Data loading and preprocessing tests
- Configuration verification
- Shape and type checking

## Docker Deployment

### Build Docker Image

```bash
make docker-build
# or
docker build -t cifar10-classifier .
```

### Run Docker Container

```bash
make docker-run
# or
docker run -p 8000:8000 cifar10-classifier
```

## Project Structure

```
cifar10-cnn-classifier/
├── source/
│   ├── __init__.py
│   ├── main.py              # Original notebook functionality
│   ├── train.py             # Enhanced training script
│   ├── config.py            # Configuration settings
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn_model.py     # Model architectures
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── data_loader.py   # Data loading utilities
│   │   └── visualization.py # Plotting functions
│   └── api/
│       ├── __init__.py
│       ├── app.py           # FastAPI application
│       └── schemas.py       # Pydantic schemas
├── tests/
│   ├── __init__.py
│   └── test_model.py        # Unit tests
├── sample/
│   ├── plots/               # Generated plots
│   └── images/              # Sample images
├── data/                    # Dataset cache
├── checkpoints/             # Model checkpoints
├── logs/                    # Training logs
├── requirements.txt         # Dependencies
├── requirements-dev.txt     # Dev dependencies
├── Makefile                 # Convenience commands
├── README.md               # This file
├── Dockerfile              # Docker configuration
└── .gitignore              # Git ignore rules
```

## Future Improvements

- [ ] Add data augmentation using `ImageDataGenerator` or `tf.image` transforms
- [ ] Experiment with transfer learning (ResNet, EfficientNet)
- [ ] Tune hyperparameters: optimizer, learning-rate schedule, batch size, dropout, weight decay
- [ ] Convert to TensorFlow `tf.data` pipeline for better performance
- [ ] Add model versioning and A/B testing support
- [ ] Implement model monitoring and logging
- [ ] Add support for custom datasets
- [ ] Create web UI for interactive predictions

## Performance Tips

- **GPU Training**: Use a GPU (Google Colab or local CUDA setup) for faster training
- **Batch Size**: Increase batch size if you have more GPU memory
- **Data Augmentation**: Enable augmentation for better generalization
- **Learning Rate**: Tune learning rate for optimal convergence
- **Model Architecture**: Experiment with deeper networks or residual connections

## Notes

- This repository ships code that will download the dataset on first run and cache it in the Keras cache directory `~/.keras/datasets/`
- Training on CPU may be slow; use GPU (Colab or a local machine with CUDA) for faster results
- The API automatically loads the best model checkpoint on startup
- All plots and logs are saved automatically during training

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{cifar10_cnn_classifier,
  title = {CIFAR-10 CNN Classifier},
  author = {Your Name},
  year = {2024},
  url = {https://github.com/yourusername/cifar10-cnn-classifier}
}
```

## Acknowledgments

- CIFAR-10 dataset: [Learning Multiple Layers of Features from Tiny Images](https://www.cs.toronto.edu/~kriz/learning-features-2009-TR.pdf), Alex Krizhevsky, 2009
- TensorFlow/Keras documentation and tutorials
- FastAPI documentation

## Support

For questions or issues, please open an issue on GitHub or contact [your-email@example.com]
