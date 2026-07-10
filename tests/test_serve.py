from fastapi import responses
from fastapi.testclient import TestClient
from src.serve import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_predict():
    payload = {
        "MedInc": 8.3, "HouseAge": 41.0, "AveRooms": 6.98,
        "AveBdrms": 1.02, "Population": 322.0, "AveOccup": 2.55,
        "Latitude": 37.88, "Longitude": -122.23
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_price" in response.json()