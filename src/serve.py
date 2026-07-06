from typing import Any


from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

app = FastAPI()

with open("models/models.pkl", "rb") as f:
    pipeline = joblib.load(f)

class Housefeatures(BaseModel):
    MedInc: float
    HouseAge: float
    AveRooms: float
    AveBdrms: float
    Population: float
    AveOccup: float
    Latitude: float
    Longitude: float


@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(features: Housefeatures):
    data = np.array(list(features.model_dump().values())).reshape(1, -1)
    prediction = pipeline.predict(data)
    return {"Predicted_ price": round(float(prediction[0]), 4)}