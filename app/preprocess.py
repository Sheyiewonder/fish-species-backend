import numpy as np
from PIL import Image

IMAGE_SIZE = (224, 224)


def preprocess_image(image: Image.Image):
    """
    Resize and normalize an uploaded image
    so it can be passed into MobileNetV2.
    """

    image = image.convert("RGB")

    image = image.resize(IMAGE_SIZE)

    image = np.array(image, dtype=np.float32)

    image /= 255.0

    image = np.expand_dims(image, axis=0)

    return image