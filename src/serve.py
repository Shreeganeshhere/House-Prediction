from typing import Any
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np
from contextlib import asynccontextmanager
import time
from prometheus_fastapi_instrumentator import Instrumentator

ml_model = {}
prediction_log = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    with open("models/models.pkl", "rb") as f:
        ml_model["pipeline"] = joblib.load(f)
    print("Model loaded successfully")
    
    yield
    
    # Shutdown
    ml_model.clear()
    prediction_log.clear()
    print("Model unloaded")

app = FastAPI(lifespan=lifespan)

Instrumentator().instrument(app).expose(app)
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
    """
    Returns ok if the API is running and the model is loaded
    """
    return {"status": "ok"}

@app.post("/predict")
def predict(features: Housefeatures):
    """
    Accepts House features and returns the predicted price

    Args:
        features (Housefeatures): House features

    Returns: The predicted price and latency seconds
    """
    start = time.time()

    data = np.array([[*features.model_dump().values()]])
    prediction = float(ml_model["pipeline"].predict(data)[0])

    latency = round(time.time() - start, 4)
    prediction_log.append({
        "prediction": round(prediction, 4),
        "latency_seconds": latency
    })

    return {"predicted_price": round(prediction, 4),
        "latency_seconds": latency
    }

@app.get("/metrics/summary")
def metrics_summary():
    """
    Returns aggregated stats on predictions served since last startup
    """
    if not prediction_log:
        return {"message": "No predictions yet"}
    
    predictions = [p["prediction"] for p in prediction_log]
    latencies   = [p["latency_seconds"] for p in prediction_log]
    
    return {
        "total_predictions": len(prediction_log),
        "avg_prediction": round(sum(predictions) / len(predictions), 4),
        "min_prediction": round(min(predictions), 4),
        "max_prediction": round(max(predictions), 4),
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 4),
    }
