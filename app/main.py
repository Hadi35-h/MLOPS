from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.predict import Predictor
from src.utils import setup_logger

logger = setup_logger("config/config.yaml", name="FastAPIApp")

# إنشاء كائن التطبيق مع تفعيل التوثيق التلقائي
app = FastAPI(
    title="Olist Delivery Prediction API",
    description="API لتوقع تأخير شحنات طلبات Olist",
    version="1.0.0",
    docs_url="/docs",  # مسار التوثيق التفاعلي
    redoc_url="/redoc",
)

# تحميل كائن التوقع عند بدء التطبيق
predictor = Predictor()


# مخطط البيانات المدخلة
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
def health_check():
    return {"status": "healthy", "message": "Olist Prediction API is running!"}


@app.post("/predict")
def predict_delivery(order: OrderInput):
    try:
        data = order.model_dump()
        result = predictor.predict(data)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"خطأ أثناء معالجة الطلب: {str(e)}")
        raise HTTPException(status_code=500, detail=f"فشلت عملية التوقع: {str(e)}")
