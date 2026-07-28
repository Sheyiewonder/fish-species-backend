# from pathlib import Path

# import numpy as np
# from tensorflow.keras.models import load_model

# from .labels import CLASS_NAMES

# MODEL_PATH = Path("app/model/best_fish_classifier.keras")

# print("Loading model...")

# model = load_model(MODEL_PATH)

# print("Model loaded successfully.")
# print(MODEL_PATH.resolve())
# model.summary()


# def predict(image_array):
#     """
#     Run inference on a preprocessed image.
#     """

#     predictions = model.predict(image_array, verbose=0)[0]

#     print("\nPrediction probabilities:")
#     for label, prob in zip(CLASS_NAMES, predictions):
#         print(f"{label}: {prob:.4f}")

#     index = int(np.argmax(predictions))

#     return CLASS_NAMES[index], float(predictions[index] * 100)
    
#     print("Raw predictions:", predictions)
#     print("Predicted index:", np.argmax(predictions))
#     print("Predicted label:", CLASS_NAMES[np.argmax(predictions)])

#     index = int(np.argmax(predictions))

#     species = CLASS_NAMES[index]

#     accuracy = float(predictions[0][index] * 100)

#     return species, accuracy

from pathlib import Path

import numpy as np
from tensorflow.keras.models import load_model

from .labels import CLASS_NAMES

MODEL_PATH = Path("app/model/best_fish_classifier.keras")

print("Loading model...")

model = load_model(MODEL_PATH)

print("Model loaded successfully.")


# -----------------------------
# Thresholds
# -----------------------------
CONFIDENCE_THRESHOLD = 30.0   # %
MARGIN_THRESHOLD = 10.0       # %


def predict(image_array):
    """
    Predict fish species or return Unknown
    if the model is not confident enough.
    """

    predictions = model.predict(image_array, verbose=0)[0]

    # Sort probabilities (highest first)
    sorted_indices = np.argsort(predictions)[::-1]

    best_index = sorted_indices[0]
    second_index = sorted_indices[1]

    best_probability = float(predictions[best_index] * 100)
    second_probability = float(predictions[second_index] * 100)

    margin = best_probability - second_probability

    print("\nPrediction probabilities:")

    for i, probability in enumerate(predictions):
        print(f"{CLASS_NAMES[i]}: {probability:.4f}")

    print(f"\nBest Confidence : {best_probability:.2f}%")
    print(f"Second Confidence: {second_probability:.2f}%")
    print(f"Margin: {margin:.2f}%")

    # Reject uncertain predictions
    if (
        best_probability < CONFIDENCE_THRESHOLD
        or margin < MARGIN_THRESHOLD
    ):
        return "Unknown", best_probability

    return CLASS_NAMES[best_index], best_probability