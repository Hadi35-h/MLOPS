from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
from src.predict import Predictor

app = FastAPI()
predictor = Predictor()


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


@app.get("/health")
def health_check():
    return {"status": "ok"}


@app.post("/predict")
def predict(data: OrderInput):
    try:
        df = pd.DataFrame([data.model_dump()])
        res = predictor.predict(df)

        # استخراج القيم بشكل فردي إذا كان العائد Tuple
        if isinstance(res, tuple):
            pred, prob = res
            return {
                "prediction": int(pred[0]) if hasattr(pred, "__len__") else int(pred),
                "probability": (
                    float(prob[0]) if hasattr(prob, "__len__") else float(prob)
                ),
            }

        return {"prediction": res.tolist()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
