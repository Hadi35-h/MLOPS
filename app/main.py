import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
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
    predictor = None


class OrderInput(BaseModel):
    total_price: float
    total_freight: float
    total_items: int
    total_payment: float
    max_installments: int
    order_status: str
    order_approved_at: str
    order_delivered_carrier_date: str
    customer_state: str


@app.get("/")
def read_root():
    return {"message": "API is up and running"}


@app.post("/predict")
def predict_endpoint(order: OrderInput):
    if predictor is None:
        raise HTTPException(status_code=500, detail="Predictor fail to load model.")

    try:
        data = order.model_dump() if hasattr(order, "model_dump") else order.dict()
        result = predictor.predict(data)
        return result
    except Exception as e:
        # إرجاع نص الاستثناء بدقة بدلاً من رمي خطأ مبهم
        raise HTTPException(status_code=500, detail=f"Prediction Exception: {str(e)}")
