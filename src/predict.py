import numpy as np
from PIL import Image

class_names = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def preprocess_image(image: Image.Image):
    image = image.resize((32, 32))
    image = np.array(image) / 255.0
    image = np.expand_dims(image, axis=0)
    return image

def predict_image(model, image: Image.Image):
    image = preprocess_image(image)
    predictions = model(image)
    predicted_class = np.argmax(predictions)
    return class_names[predicted_class]