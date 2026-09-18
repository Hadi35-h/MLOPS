import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException
import pandas as pd
from pydantic import BaseModel, field_validator, Field
from src.predict import Predictor

app = FastAPI()
predictor = None
logger = logging.getLogger("FastAPIApp")

# 1. Set up logging directory
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / "prediction_logs.jsonl"


def load_predictor():
    """Load the Predictor with model and preprocessor artifacts."""
    global predictor
    if predictor is not None:
        return
    try:
        predictor = Predictor()
    except Exception as e:
        logger.error(f"Failed to load predictor: {e}")
        predictor = None


# Attempt to load models at startup (non-fatal if models are missing)
load_predictor()


class OrderInput(BaseModel):
    total_price: float = Field(..., gt=0)
    total_freight: float = Field(..., ge=0)
    total_items: int = Field(..., ge=1)
    total_payment: float = Field(..., gt=0)
    max_installments: int = Field(..., ge=0)
    order_status: str = Field(..., min_length=1)
    order_approved_at: str = Field(..., min_length=1)
    order_delivered_carrier_date: str = Field(..., min_length=1)
    customer_state: str = Field(..., min_length=2, max_length=2)

    @field_validator("order_approved_at", "order_delivered_carrier_date")
    @classmethod
    def validate_date_format(cls, v):
        from datetime import datetime as dt

        try:
            dt.fromisoformat(v)
        except ValueError:
            raise ValueError("Invalid ISO format datetime string")
        return v


def log_prediction(input_data: dict, prediction_result: dict):
    """Helper function to save input and prediction to a JSONL log file"""
    log_entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "input": input_data,
        "output": prediction_result,
        # Left empty until matched with the actual delivery date later
        "actual_delivered_customer_date": None,
    }
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")


@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": predictor is not None}


@app.post("/predict")
def predict(data: OrderInput):
    # Reload models if predictor is None
    if predictor is None:
        load_predictor()

    if predictor is None:
        raise HTTPException(
            status_code=500,
            detail="Predictor model files not found or failed to load.",
        )

    try:
        input_dict = data.model_dump()
        df = pd.DataFrame([input_dict])
        res = predictor.predict(df)

        response_data = {}

        # 2. Extract and build the result
        if isinstance(res, tuple):
            pred, prob = res
            response_data = {
                "prediction": (int(pred[0]) if hasattr(pred, "__len__") else int(pred)),
                "probability": (
                    float(prob[0]) if hasattr(prob, "__len__") else float(prob)
                ),
            }
        elif isinstance(res, dict):
            response_data = res
        else:
            response_data = {"prediction": res.tolist()}

        # 3. Log the input and prediction to the log file
        log_prediction(input_dict, response_data)

        return response_data

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
