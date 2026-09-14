import logging
from pathlib import Path
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
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
    total_price: float = Field(..., gt=0, description="يجب أن يكون السعر أكبر من 0")
    total_freight: float = Field(
        ..., ge=0, description="تكلفة الشحن لا يمكن أن تكون سالبة"
    )
    total_items: int = Field(
        ..., gt=0, description="عدد العناصر يجب أن يكون 1 على الأقل"
    )
    total_payment: float = Field(
        ..., gt=0, description="المبلغ المدفوع يجب أن يكون أكبر من 0"
    )
    max_installments: int = Field(
        ..., ge=1, description="عدد الأقساط يجب أن يكون 1 على الأقل"
    )

    order_status: str
    order_approved_at: str
    order_delivered_carrier_date: str

    # يفرض إدخال حرفين كبيرين بالضبط (مثل SP أو RJ)
    customer_state: str = Field(
        ...,
        min_length=2,
        max_length=2,
        pattern="^[A-Z]{2}$",
        description="رمز الولاية يجب أن يتكون من حرفين كبيرين بالإنجليزية فقط",
    )


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
