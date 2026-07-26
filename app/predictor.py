from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from .labels import CLASS_NAMES

MODEL_PATH = Path("app/model/best_fish_classifier.keras")

print("Loading model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")


def predict(image_array):
    """
    Run inference on a preprocessed image.
    """

    predictions = model.predict(image_array, verbose=0)

    index = int(np.argmax(predictions))

    species = CLASS_NAMES[index]

    accuracy = float(predictions[0][index] * 100)

    return species, accuracy