from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from src.predict import Predictor
from src.utils import setup_logger

logger = setup_logger("config/config.yaml", name="FastAPIApp")

app = FastAPI(
    title="Olist Delivery Prediction API",
    description="API لتوقع تأخير شحنات طلبات Olist",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

predictor = Predictor()


# مخطط البيانات المدخلة مع تفعيل القيود الـ Validation
class OrderInput(BaseModel):
    total_price: float = Field(
        ..., ge=0, description="السعر الاجمالي يجب أن يكون أكثر أو يساوي صفر"
    )
    total_freight: float = Field(..., ge=0, description="تكلفة الشحن")
    total_items: int = Field(..., ge=1, description="عدد القطع يجب أن يكون 1 على الأقل")
    total_payment: float = Field(..., ge=0, description="قيمة الدفع")
    max_installments: int = Field(..., ge=1, description="عدد الأقساط")
    order_status: str = Field(..., example="delivered")
    order_approved_at: str = Field(..., example="2017-09-03 08:43:00")
    order_delivered_carrier_date: str = Field(..., example="2017-09-03 21:16:00")
    customer_state: str = Field(..., min_length=2, max_length=2, example="SP")


@app.get("/")
def health_check():
    return {"status": "healthy", "message": "Olist Prediction API is running!"}


@app.get("/health")
def status_health():
    # مسار health check إضافي مطابقة لمتطلبات التاسك
    return {"status": "healthy", "model_version": "1.0.0"}


@app.post("/predict")
def predict_delivery(order: OrderInput):
    try:
        data = order.model_dump()
        result = predictor.predict(data)
        logger.info(f"Successful prediction: {result}")
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"خطأ أثناء معالجة الطلب: {str(e)}")
        raise HTTPException(status_code=500, detail=f"فشلت عملية التوقع: {str(e)}")
