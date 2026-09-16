import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException

# استيراد الـ Schema من الملف الجديد
from app.schemas import OrderInput, PredictionOutput
from src.predict import Predictor

app = FastAPI(title="MLOps Prediction API")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"

try:
    predictor = Predictor(
        model_path=str(MODEL_PATH), preprocessor_path=str(PREPROCESSOR_PATH)
    )
except Exception as e:
    logging.error(f"Failed to load predictor models: {e}")
    predictor = None


@app.get("/")
def read_root():
    return {"message": "Welcome to MLOps API"}


@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict_endpoint(order: OrderInput):
    if predictor is None:
        raise HTTPException(
            status_code=500, detail="Predictor model files not found or failed to load."
        )

    try:
        data = order.model_dump() if hasattr(order, "model_dump") else order.dict()
        res = predictor.predict(data)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
