# 📦 Olist Late Delivery Prediction - MLOps Pipeline

مشروع MLOps شامل لتحويل نوت بوك التنبؤ بتأخير شحنات طلبات **Olist** إلى خدمة إنتاجية حية (Inference Pipeline) قابلة للتوسع والمراقبة.

---

## 🛠️ هيكلية المشروع (Project Structure)

```text
├── app/             # تطبيق FastAPI والمسارات (Endpoints)
├── artifacts/       # ملفات النموذج والمحولات المحفوظة والـ Logs
├── config/          # ملف الإعدادات والمسارات الديناميكية (yaml)
├── data/            # البيانات الخام والمعالجة
├── src/             # كود البايثون الموديولار (Preprocessing & Prediction)
├── tests/           # اختبارات الوحدة (Unit) والتكامل (Integration)
├── Dockerfile       # ملف بناء الحاوية
└── docker-compose.yml # تشغيل النظام كاملاً
