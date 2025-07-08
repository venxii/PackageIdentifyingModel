from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import uvicorn
import json
import os
from PIL import Image
import io

app = FastAPI()

# Allow frontend calls
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model and labels
MODEL_PATH = "packaging_classifier.h5"
LABELS_PATH = "labels.json"
IMG_SIZE = (224, 224)

model = load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    class_names = json.load(f)

# Simple rule-based logic for demo
RECYCLABLE_CLASSES = ["plastic", "cardboard", "glass", "metal"]
IMPACT_SCORES = {
    "plastic": "high",
    "glass": "low",
    "cardboard": "low",
    "metal": "medium",
    "other": "unknown"
}

@app.get("/")
def root():
    return {"status": "API is running."}

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    img_bytes = await file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    preds = model.predict(img_array)
    pred_class = class_names[np.argmax(preds)]

    return {
        "class": pred_class,
        "recyclable": pred_class in RECYCLABLE_CLASSES,
        "co2_impact": IMPACT_SCORES.get(pred_class, "unknown")
    }

