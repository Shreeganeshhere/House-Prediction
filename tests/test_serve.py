from fastapi.testclient import TestClient
from unittest.mock import MagicMock, patch
import numpy as np

# Mock the model before importing app
mock_pipeline = MagicMock()
mock_pipeline.predict.return_value = np.array([2.5])

with patch("builtins.open"), patch("pickle.load", return_value=mock_pipeline):
    from src.serve import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict():
    payload = {
        "MedInc": 8.3, "HouseAge": 41.0, "AveRooms": 6.98,
        "AveBedrms": 1.02, "Population": 322.0, "AveOccup": 2.55,
        "Latitude": 37.88, "Longitude": -122.23
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_price" in response.json()
    assert "latency_seconds" in response.json()

def test_metrics_summary_empty():
    response = client.get("/metrics/summary")
    assert response.status_code == 200