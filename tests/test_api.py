from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/")
    assert response.status_code == 200


def test_predict_endpoint():
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
    assert response.status_code == 200

    # التعديل هنا الوصول لمفتاح is_late داخل data
    res_json = response.json()
    assert "data" in res_json
    assert "is_late" in res_json["data"]
