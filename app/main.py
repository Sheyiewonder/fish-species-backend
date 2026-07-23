from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .predictor import predict
from .preprocess import preprocess_image

app = FastAPI(title="Fish Species Identifier API")

# Allow requests from the React frontend during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "https://fish-species-frontend-bice.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {
        "message": "Fish Species Identifier API is running."
    }


@app.post("/predict")
async def predict_species(file: UploadFile = File(...)):
    """
    Receive an uploaded image and return
    the predicted fish species.
    """

    image_bytes = await file.read()

    image = Image.open(BytesIO(image_bytes))

    processed = preprocess_image(image)

    result = predict(processed)

    return result
