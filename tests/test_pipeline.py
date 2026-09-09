import pytest
import pandas as pd
from src.predict import Predictor


def test_predictor_initialization():
    predictor = Predictor()
    assert predictor is not None


def test_predictor_execution():
    predictor = Predictor()
    sample_data = pd.DataFrame(
        [
            {
                "total_price": 100.0,
                "total_freight": 20.0,
                "total_items": 1,
                "total_payment": 120.0,
                "max_installments": 1,
                "order_status": "delivered",
                "order_approved_at": "2026-09-01T10:00:00",
                "order_delivered_carrier_date": "2026-09-02T14:30:00",
                "customer_state": "SP",
            }
        ]
    )

    res = predictor.predict(sample_data)
    assert res is not None

    # التحقق من إرجاع Tuple يحتوي على النتيجة والإحتمالية
    if isinstance(res, tuple):
        pred, prob = res
        assert pred is not None
        assert prob is not None
    else:
        assert len(res) == 1
