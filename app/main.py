from io import BytesIO

from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .predictor import predict
from .preprocess import preprocess_image
from .species_info import SPECIES_INFO

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

    species, accuracy = predict(processed)

    info = SPECIES_INFO.get(species)

    if info is None:
        return {
            "species": species,
            "accuracy": accuracy,
            "scientific_name": "",
            "description": "No information available for this species.",
            "features": [],
            "habitat": "",
            "diet": "",
            "average_length": "",
        }

    return {    
        "species": species,
        "scientific_name": info["scientific_name"],
        "accuracy": accuracy,
        "features": info["features"],
        "habitat": info["habitat"],
        "diet": info["diet"],
        "average_length": info["average_length"],
        "description": info["description"],
    }
