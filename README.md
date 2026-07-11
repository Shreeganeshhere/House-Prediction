# House Price Predictor — MLOps Pipeline

An end-to-end MLOps project built on the California Housing dataset, covering data versioning, experiment tracking, model serving, cloud deployment, and CI/CD automation.

---

## Stack

| Layer | Tool |
|---|---|
| Data versioning | DVC + GCS |
| Experiment tracking | MLflow |
| ML pipeline | Sklearn Pipeline + XGBoost |
| Serving | FastAPI + Uvicorn |
| Containerization | Docker + Artifact Registry |
| Cloud deployment | GCP Cloud Run |
| CI/CD | GitHub Actions + Workload Identity Federation |

---

## Project Structure

```
house-price-mlops/
├── data/
│   ├── raw/                  # DVC-tracked raw CSV
│   └── processed/            # DVC-tracked train/test splits
├── src/
│   ├── data_ingestion.py     # Fetches California Housing dataset
│   ├── transform.py          # Train/test split
│   ├── train.py              # XGBoost + Sklearn Pipeline + MLflow
│   └── serve.py              # FastAPI inference endpoint
├── models/                   # DVC-tracked model artifact
├── tests/
│   └── test_serve.py         # FastAPI endpoint tests
├── .github/
│   └── workflows/
│       └── deploy.yml        # CI/CD pipeline
├── dvc.yaml                  # Pipeline stage definitions
├── params.yaml               # Hyperparameters
├── metrics.json              # Latest run metrics
├── Dockerfile
└── requirements.txt
```

---

## Pipeline Phases

### Phase 1 — Data Versioning
DVC tracks raw data with GCS as remote storage. Every data version is reproducible via `dvc pull`.

### Phase 2 — Feature Engineering & Training
Modular scripts wired as DVC pipeline stages. `dvc repro` reruns only changed stages.

### Phase 3 — Experiment Tracking
MLflow autologging captures params, metrics, model artifacts, and signatures on every run. Compare runs at `http://localhost:5000`.

### Phase 4 — Model Serving
FastAPI exposes `/predict` and `/health` endpoints. Pydantic validates input schema. Containerized with Docker.

### Phase 5 — Cloud Deployment
Docker image pushed to GCP Artifact Registry. Deployed serverlessly on Cloud Run with HTTPS auto-provisioned.

### Phase 6 — CI/CD
GitHub Actions triggers on every push to `main`: pulls data → reruns pipeline → runs tests → builds and pushes image → deploys to Cloud Run. Auth via Workload Identity Federation — no keys stored anywhere.

---

## Quickstart

```bash
git clone https://github.com/YOUR_USERNAME/house-price-mlops
cd house-price-mlops
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
dvc pull
dvc repro
```

### Run locally
```bash
uvicorn src.serve:app --reload
```

### Test prediction
```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "MedInc": 8.3,
    "HouseAge": 41.0,
    "AveRooms": 6.98,
    "AveBedrms": 1.02,
    "Population": 322.0,
    "AveOccup": 2.55,
    "Latitude": 37.88,
    "Longitude": -122.23
  }'
```

### Run tests
```bash
pytest tests/ -v
```

---

## Metrics

| Metric | Value |
|---|---|
| RMSE | see `metrics.json` |
| R² | see `metrics.json` |

---

## CI/CD Flow

```
push to main
    → dvc pull          (fetch data from GCS)
    → dvc repro         (retrain if anything changed)
    → pytest            (validate endpoints)
    → docker build      (package model + API)
    → docker push       (Artifact Registry)
    → gcloud run deploy (Cloud Run — live HTTPS endpoint)
```

Auth via GCP Workload Identity Federation — no service account keys stored in GitHub secrets.

---

## API Reference

### `GET /health`
```json
{ "status": "ok" }
```

### `POST /predict`
**Request:**
```json
{
  "MedInc": 8.3,
  "HouseAge": 41.0,
  "AveRooms": 6.98,
  "AveBedrms": 1.02,
  "Population": 322.0,
  "AveOccup": 2.55,
  "Latitude": 37.88,
  "Longitude": -122.23
}
```
**Response:**
```json
{ "predicted_price": 4.526 }
```