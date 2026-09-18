import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Dynamically add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from app.main import app

client = TestClient(app)


def test_read_root():
    """Test the root path"""
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "API is up and running"}


def test_predict_endpoint():
    """Test prediction and show error details if the call fails"""
    payload = {
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
    response = client.post("/predict", json=payload)

    # Print error response immediately if not returning 200
    if response.status_code != 200:
        print("\n[SERVER ERROR DETAILED RESPONSE]:", response.text)

    assert response.status_code == 200
    assert "prediction" in response.json()
