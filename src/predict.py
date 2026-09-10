import joblib
import pandas as pd
import yaml
from src.utils import logger


class Predictor:
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, "r", encoding="utf-8") as f:
            self.config = yaml.safe_load(f)

        # تحميل الموديل والـ Preprocessor
        self.preprocessor = joblib.load(self.config["paths"]["preprocessor_path"])
        self.model = joblib.load(self.config["paths"]["model_path"])
        self.threshold = self.config["model_params"]["threshold"]
        logger.info("تم تحميل الموديل والـ Preprocessor بنجاح.")

    def predict(self, data: dict):
        # تحويل القاموس إلى DataFrame بحجم عينة واحدة (1 Row, N Columns)
        df = pd.DataFrame([data])

        # تحويل الأعمدة إذا كانت تحتاج لمعالجة تواريخ قبل المعالج
        if "order_approved_at" in df.columns:
            df["order_approved_at"] = pd.to_datetime(df["order_approved_at"])
        if "order_delivered_carrier_date" in df.columns:
            df["order_delivered_carrier_date"] = pd.to_datetime(
                df["order_delivered_carrier_date"]
            )

        # التوقع باستخدام الـ Pipeline أو Model
        prediction = self.model.predict(df)

        # إرجاع النتيجة
        return {"prediction": int(prediction[0])}
