import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException

from app.schemas import OrderInput, PredictionOutput
from src.predict import Predictor

app = FastAPI(title="MLOps Prediction API")

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"

predictor = None


def load_models():
    """دالة تحكم لإعادة تحميل النماذج ديناميكياً من الذاكرة/الملفات"""
    global predictor
    try:
        predictor = Predictor(
            model_path=str(MODEL_PATH), preprocessor_path=str(PREPROCESSOR_PATH)
        )
    except Exception as e:
        logging.error(f"Failed to load predictor models: {e}")
        predictor = None


# تحميل النماذج عند بدء التطبيق
load_models()


@app.get("/")
def read_root():
    return {"message": "API is up and running"}


@app.get("/healthcheck")
def healthcheck():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict_endpoint(order: OrderInput):
    # محاولة إعادة التحميل إذا كان الكائن None قبل إرجاع الخطأ
    if predictor is None:
        load_models()

    if predictor is None:
        raise HTTPException(
            status_code=500,
            detail="Predictor model files not found or failed to load.",
        )

    try:
        data = (
            order.model_dump() if hasattr(order, "model_dump") else order.dict()
        )
        res = predictor.predict(data)
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))