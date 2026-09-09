import pytest
from fastapi.testclient import TestClient
from app.main import app

# إنشاء عميل وهمي لاختبار نقاط النهاية الخاصة بـ FastAPI
client = TestClient(app)


def test_health_endpoint():
    """فحص مسار صحة الخدمة للتأكد من أنها تعمل."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_pipeline_success():
    """فحص دورة التوقع الكاملة عند إرسال مدخلات صحيحة."""
    payload = {
        "order_item_id": 1,
        "price": 100.0,
        "freight_value": 15.0,
        "payment_sequential": 1,
        "payment_installments": 2,
        "payment_value": 115.0,
    }
    response = client.post("/predict", json=payload)

    # 1. التأكد من نجاح الاستجابة
    assert response.status_code == 200

    # 2. التأكد من وجود الحقول الأساسية في النتائج
    data = response.json()
    assert "prediction" in data
    assert "probability" in data

    # 3. التأكد من أن قيمة الاحتمالية منطقية (بين 0 و 1)
    assert 0.0 <= data["probability"] <= 1.0


def test_predict_pipeline_invalid_input():
    """فحص قدرة الـ Pipeline على الرفض عند إرسال أسعار بالسالب."""
    payload = {
        "order_item_id": 1,
        "price": -50.0,  # قيمة خاطئة
        "freight_value": 15.0,
        "payment_sequential": 1,
        "payment_installments": 2,
        "payment_value": 115.0,
    }
    response = client.post("/predict", json=payload)

    # يجب أن ترفض الخدمة الطلب وتُرجع كود 400 Bad Request
    assert response.status_code == 400
