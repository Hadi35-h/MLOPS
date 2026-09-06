from src.predict import Predictor
from src.utils import setup_logger

logger = setup_logger("PipelineTest")

# 1. إعداد عينة بيانات مدخلة (Dummy Sample)
# ملاحظة: استبدل الميزات بأسمائها الحقيقية التي استخدمتها في النوت بوك الخامسة
# عينة مدخلات تجريبية تحتوي على جميع الأعمدة المطلوبة للـ Preprocessor
dummy_input = {
    "total_price": 120.0,
    "total_freight": 18.5,
    "total_items": 1,
    "total_payment": 138.5,
    "max_installments": 2,
    "order_status": "delivered",
    "order_approved_at": "2023-01-01 10:00:00",
    "order_delivered_carrier_date": "2023-01-02 14:00:00",
    "customer_state": "SP",  # إضافة ولاية العميل
}


def run_test():
    try:
        logger.info("بدء اختبار بايبلاين التوقع...")

        # 2. تحميل كائن التوقع
        predictor = Predictor()

        # 3. تشغيل عملية التوقع
        result = predictor.predict(dummy_input)

        logger.info("--- نتيجة الاختبار ---")
        logger.info(f"النتيجة المرجعة: {result}")
        print("\n✅ نجح الاختبار! النظام جاهز لاستقبال الطلبات.")

    except Exception as e:
        logger.error(f"❌ فشل الاختبار بسبب الخطأ التالي: {str(e)}")


if __name__ == "__main__":
    run_test()
