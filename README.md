# 📦 Olist Late Delivery Prediction — MLOps Pipeline

<a href="https://github.com/Hadi35-h/MLOPS"><img src="https://img.shields.io/badge/license-MIT-blue.svg" alt="License"></a>
<img src="https://img.shields.io/badge/python-3.11+-brightgreen" alt="Python">
<img src="https://img.shields.io/badge/fastapi-0.115+-blue" alt="FastAPI">
<img src="https://img.shields.io/badge/mlflow-3.16-orange" alt="MLflow">
<img src="https://img.shields.io/badge/sklearn-1.9-red" alt="Scikit-learn">

## 🔗 Quick Links

- [GitHub Repository](https://github.com/Hadi35-h/MLOPS)
- [API Documentation (Swagger UI)](http://localhost:8000/docs)
- [API Reference (Redoc)](http://localhost:8000/redoc)
- [MLflow Tracking UI](http://localhost:5000)
- [Evidently Drift Monitor](http://localhost:8080)
- [Docker Hub Images](https://hub.docker.com/)

---

## 🎯 Project Overview

**Olist Late Delivery Prediction** هو مشروع **MLOps** شامل يحوّل نوت بوك التنبؤ بتأخير شحنات طلبات **Olist** إلى خدمة إنتاجية حية (Inference Pipeline) قائمة على **FastAPI**، قابلة للتوسع والمراقبة. يدمج الـ **ML pipeline** الكامل من جمع البيانات وحتى النشر والمراقبة (MLOps):

- **Data Validation** — فحص البيانات المدخلة قبل الاستخدام.
- **Preprocessing** — تنظيف وتجهيز البيانات (التعامل مع القيم الناقصة، تحويل التواريخ، ترميز الفئات).
- **Model Training** — تدريب نموذج RandomForestClassifier مع تسجيل التجارب في MLflow.
- **Model Serving** — خدمة REST API عبر FastAPI / Uvicorn لإرجاع التنبؤات في الوقت الحقيقي.
- **Monitoring** — اكتشاف Data Drift باستخدام Evidently.
- **CI/CD** — اختبارات أوتوماتيكية وتوثيق برمجيات.

---

## 📋 Table of Contents

1. [Project Structure](#-project-structure)
2. [ML Pipeline Architecture](#-ml-pipeline-architecture)
3. [Data Schema](#-data-schema)
4. [Installation](#-installation)
5. [Usage](#-usage)
6. [API Endpoints](#-api-endpoints)
7. [Testing](#-testing)
8. [Docker Deployment](#-docker-deployment)
9. [MLflow Tracking](#-mlflow-tracking)
10. [Data Drift Monitoring](#-data-drift-monitoring)
11. [Configuration](#-configuration)
12. [Pre-commit Hooks](#-pre-commit-hooks)

---

## 🗂️ Project Structure

```text
olist-mlops/
├── app/
│   ├── main.py          # FastAPI application — endpoints: /, /healthcheck, /predict
│   └── schemas.py       # Pydantic models — OrderInput, PredictionOutput
├── src/
│   ├── __init__.py
│   ├── preprocessing.py # Data cleaning & preprocessing functions
│   ├── data_validator.py# Input data validation (negative values, missing data)
│   ├── train.py          # Training pipeline — RandomForest + MLflow logging
│   ├── predict.py        # Inference pipeline — loads model & preprocessor, runs prediction
│   ├── main.py           # Alternative FastAPI app — endpoints: /health, /predict
│   ├── mlflow_tracker.py # Registers model + preprocessor + config as MLflow artifacts
│   ├── monitoring.py     # Data drift report generation (Evidently DataDriftPreset)
│   ├── evidently_report.py # Evidently report with random reference/current data
│   └── utils.py          # Logging setup (file + console)
├── tests/
│   ├── test_api.py       # Integration tests for src/main.py (/health, /predict)
│   └── test_pipeline.py  # Integration tests for app/main.py (/, /predict)
├── config/
│   └── config.yaml       # Paths, model info, data schema, model params
├── data/
│   ├── raw/              # Raw Olist datasets
│   └── processed/        # Train/test splits
├── models/
│   ├── model.pkl         # Trained model (joblib)
│   └── preprocessor.pkl  # Preprocessing pipeline (ColumnTransformer, joblib)
├── artifacts/            # MLflow artifacts, logs, drift reports
├── notebooks/            # Exploratory analysis notebooks
├── requirements.txt      # All Python dependencies
├── Dockerfile            # Container image build
├── docker-compose.yml    # Multi-service orchestration
├── .pre-commit-config.yaml
├── .env                  # Environment variables (DB credentials)
└── README.md
```

---

## 🏗️ ML Pipeline Architecture

```
┌──────────────────────────────────────────────────────────┐
│                    Data Sources                           │
│              (Olist Orders Dataset)                        │
└─────────────────────┬──────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────┐
│              Data Validation  (src/data_validator.py)      │
│  • Check negative numeric values                          │
│  • Check missing values                                    │
└─────────────────────┬──────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────┐
│       Preprocessing  (src/preprocessing.py)                │
│  • Parse date columns                                     │
│  • Impute missing values (median for numeric,              │
│    "Unknown" for categorical)                             │
└─────────────────────┬──────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────┐
│      Feature Engineering  (src/predict.py, fallback path)   │
│  • Calculate date-based features:                         │
│    - approval_to_carrier_hours                             │
│    - order_approved_hour                                   │
│    - order_approved_dayofweek                              │
│  • One-hot encoding (pd.get_dummies)                       │
│  • Feature count alignment with model.n_features_in_       │
└─────────────────────┬──────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────┐
│        Model Training  (src/train.py)                       │
│  • RandomForestClassifier (n_estimators=100, max_depth=5)  │
│  • 80/20 train/test split                                 │
│  • Log params, metrics, and model to MLflow               │
│  • Save model.pkl locally via joblib                      │
└─────────────────────┬──────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────┐
│     Artifact Registration  (src/mlflow_tracker.py)        │
│  • Register model.pkl, preprocessor.pkl, config.yaml      │
│    as MLflow artifacts under "Olist_Inference_Pipeline"    │
└─────────────────────┬──────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────┐
│        Inference API  (app/main.py, src/predict.py)        │
│  • POST /predict → OrderInput → Predictor.predict()       │
│  • Returns {"prediction": <int>}                          │
└─────────────────────┬──────────────────────────────────────┘
                      │
┌─────────────────────▼──────────────────────────────────────┐
│      Monitoring  (src/monitoring.py, evidently_report.py)   │
│  • Generate Data Drift report (Evidently DataDriftPreset)  │
│  • Compare reference vs current data                       │
└────────────────────────────────────────────────────────────┘
```

### Pipeline Stages

| Stage | Module | Description |
|---|---|---|
| 1. Validation | `src/data_validator.py` | Validates input dict for negative numerics & missing values |
| 2. Preprocessing | `src/preprocessing.py` | Parses date columns, imputes missing values |
| 3. Training | `src/train.py` | Trains RandomForest, logs to MLflow, saves model.pkl |
| 4. Artifact Logging | `src/mlflow_tracker.py` | Registers model + preprocessor + config in MLflow |
| 5. Inference | `src/predict.py` | Loads artifacts, applies preprocessor, runs model.predict |
| 6. Serving | `app/main.py` / `src/main.py` | FastAPI endpoints for health check & prediction |
| 7. Monitoring | `src/monitoring.py` | Evidently data drift report generation |

---

## 📊 Data Schema

The API expects the following input schema (defined in `config/config.yaml` and `app/schemas.py`):

| Field | Type | Constraints | Description |
|---|---|---|---|
| `total_price` | float | gt=0 | Total order price |
| `total_freight` | float | ge=0 | Shipping cost |
| `total_items` | int | gt=0 | Number of items in order |
| `total_payment` | float | gt=0 | Total payment amount |
| `max_installments` | int | ge=1 | Number of installments |
| `order_status` | str | — | Order status (e.g. "delivered") |
| `order_approved_at` | str | — | ISO 8601 timestamp of approval |
| `order_delivered_carrier_date` | str | — | ISO 8601 timestamp of carrier delivery |
| `customer_state` | str | pattern `^[A-Z]{2}$` | 2-letter uppercase state code (e.g. "SP") |

**Output:**

```json
{
  "prediction": 1
}
```

---

## 🔧 Installation

### Prerequisites

- Python 3.11+
- pip / venv

### Steps

```bash
# 1. Clone the repository
git clone https://github.com/Hadi35-h/MLOPS.git
cd MLOPS/my_mlops_project

# 2. Create and activate a virtual environment
python -m venv venv
# Linux / macOS:
source venv/bin/activate
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Install pre-commit hooks
pip install pre-commit
pre-commit install
```

---

## 🚀 Usage

### Local API Server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

> **Note:** `app/main.py` is the primary entrypoint (defined in `Dockerfile` and `docker-compose.yml`). An alternative FastAPI app is also available at `src/main.py` (tested by `tests/test_api.py`).

### Swagger UI / Redoc

Once the server is running, visit:

- **Swagger UI:** `http://localhost:8000/docs`
- **Redoc:** `http://localhost:8000/redoc`

### Train a Model

```bash
python -m src.train
```

This will:
1. Generate synthetic data (100 samples, 10 features).
2. Train a `RandomForestClassifier`.
3. Log parameters, metrics, and the model to MLflow.
4. Save `models/model.pkl` locally.

### Log Artifacts to MLflow

```bash
python -m src.mlflow_tracker
```

### Run Prediction Directly

```python
from src.predict import Predictor

predictor = Predictor(model_path="models/model.pkl",
                      preprocessor_path="models/preprocessor.pkl")
result = predictor.predict({
    "total_price": 100.0,
    "total_freight": 20.0,
    "total_items": 1,
    "total_payment": 120.0,
    "max_installments": 1,
    "order_status": "delivered",
    "order_approved_at": "2026-09-01T10:00:00",
    "order_delivered_carrier_date": "2026-09-02T14:30:00",
    "customer_state": "SP",
})
print(result)  # {"prediction": 1}
```

---

## 🔌 API Endpoints

### `GET /` (app/main.py only)

Root endpoint.

```bash
curl http://localhost:8000/
# {"message": "API is up and running"}
```

### `GET /health` (src/main.py only)

Health check endpoint.

```bash
curl http://localhost:8000/health
# {"status": "ok"}
```

### `GET /healthcheck` (app/main.py only)

Health check with model status.

```bash
curl http://localhost:8000/healthcheck
# {"status": "ok", "model_loaded": true}
```

### `POST /predict`

Returns a delivery prediction.

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "total_price": 100.0,
    "total_freight": 20.0,
    "total_items": 1,
    "total_payment": 120.0,
    "max_installments": 1,
    "order_status": "delivered",
    "order_approved_at": "2026-09-01T10:00:00",
    "order_delivered_carrier_date": "2026-09-02T14:30:00",
    "customer_state": "SP"
  }'
# {"prediction": 1}
```

---

## 🧪 Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run a specific test file
python -m pytest tests/test_api.py -v

# Run with output capture disabled (to see print statements)
python -m pytest tests/ -v -s
```

### Test Coverage

| File | Tests | Description |
|---|---|---|
| `tests/test_api.py` | `test_health_check`, `test_predict_endpoint` | Tests for `src/main.py` (`/health`, `/predict`) |
| `tests/test_pipeline.py` | `test_read_root`, `test_predict_endpoint` | Tests for `app/main.py` (`/`, `/predict`) |

**Expected output:** `4 passed`

---

## 🐳 Docker Deployment

### Single Container (API Only)

```bash
docker build -t olist-mlops .
docker run -p 8000:8000 olist-mlops
```

### Full Stack (API + MLflow + Postgres + Evidently)

```bash
docker-compose up --build
```

This starts four services:

| Service | Port | Description |
|---|---|---|
| `postgres` | 5432 | PostgreSQL backend for MLflow |
| `mlflow` | 5000 | MLflow tracking server |
| `api` | 8000 | FastAPI inference API |
| `evidently` | 8080 | Evidently UI for drift monitoring |

### Environment Variables (docker-compose)

| Variable | Default | Description |
|---|---|---|
| `MLFLOW_TRACKING_URI` | `http://mlflow:5000` | MLflow server URL inside Docker network |

---

## 📊 MLflow Tracking

The project integrates with MLflow for experiment tracking and model registry.

### Tracking Server

- **Docker:** `http://localhost:5000`
- **Local:** `mlruns/` directory (SQLite backend by default)

### Experiments

| Experiment | Module | Logged Artifacts |
|---|---|---|
| `Olist_Churn_Prediction` | `src/train.py` | params (n_estimators, max_depth), metrics (accuracy, precision, recall, f1), model |
| `Olist_Inference_Pipeline` | `src/mlflow_tracker.py` | model.pkl, preprocessor.pkl, config.yaml |

---

## 🔍 Data Drift Monitoring

### Generate Drift Report

```python
from src.monitoring import generate_drift_report

generate_drift_report(
    reference_path="data/processed/train.csv",
    current_path="data/processed/test.csv",
    output_path="artifacts/drift_report.html"
)
```

### Evidently UI

When running via `docker-compose`, the Evidently UI is available at `http://localhost:8080`. The `src/evidently_report.py` script generates a `DataDriftPreset` report and pushes it to the Evidently workspace.

---

## ⚙️ Configuration

All paths, model info, and parameters are defined in `config/config.yaml`:

```yaml
paths:
  model_path: "artifacts/final_model.joblib"
  preprocessor_path: "artifacts/preprocessor.joblib"
  raw_data: "data/raw/olist_orders_dataset.csv"
  processed_train_data: "data/processed/train.csv"
  processed_test_data: "data/processed/test.csv"
  log_file: "artifacts/app.log"
  drift_report: "artifacts/drift_report.html"

model_info:
  name: "RandomForest_Delivery_Prediction"
  version: "1.0.0"

data_schema:
  features:
    - total_price
    - total_freight
    - total_items
    - total_payment
    - max_installments
    - order_status
    - order_approved_at
    - order_delivered_carrier_date
    - customer_state

model_params:
  threshold: 0.5
```

### Model Loading

The `Predictor` class (`src/predict.py`) loads artifacts at initialization:

- **Model default path:** `models/model.pkl`
- **Preprocessor default path:** `models/preprocessor.pkl`

Both paths can be overridden via constructor arguments.

### Fallback Preprocessing

When no preprocessor is provided, the `Predictor.predict()` method applies inline feature engineering:

1. Computes date-difference features (approval → carrier hours, approval hour, day of week).
2. One-hot encodes categorical columns (`pd.get_dummies`).
3. Aligns the feature count with `model.n_features_in_` (pads with zeros or truncates).

---

## 🔒 Pre-commit Hooks

The project uses pre-commit for code quality:

```bash
# Install hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

Configured hooks:

| Hook | Purpose |
|---|---|
| `trailing-whitespace` | Remove trailing whitespace |
| `end-of-file-fixer` | Ensure files end with a newline |
| `check-yaml` | Validate YAML syntax |
| `check-added-large-files --maxkb=10000` | Block files > 10 MB |
| `black` | Code formatting (excludes `venv/`, `artifacts/`) |

---

## 📝 License

MIT License — see `LICENSE` file for details.
