import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import Predictor

app = FastAPI()

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


@app.post("/predict")
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
