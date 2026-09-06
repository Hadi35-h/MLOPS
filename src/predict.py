import os
import joblib
import pandas as pd
from src.utils import load_config, setup_logger

logger = setup_logger("InferencePipeline")


class Predictor:

    def __init__(self, config_path: str = "config/config.yaml"):
        self.config = load_config(config_path)

        # تحميل الـ Artifacts
        preprocessor_path = self.config["paths"]["preprocessor_path"]
        model_path = self.config["paths"]["model_path"]

        logger.info(f"تحميل الـ Preprocessor من: {preprocessor_path}")
        self.preprocessor = joblib.load(preprocessor_path)

        logger.info(f"تحميل الموديل من: {model_path}")
        self.model = joblib.load(model_path)

        self.threshold = self.config["model_params"]["threshold"]

    def predict(self, raw_data: dict) -> dict:
        """استقبال dict وتطبيق المعالجة ثم التوقع"""
        # 1. تحويل المدخل إلى DataFrame
        df = pd.DataFrame([raw_data])

        # 2. تطبيق المعالجة (transform فقط من دون fit)
        processed_data = self.preprocessor.transform(df)

        # 3. حساب الاحتمالية والتوقع
        probability = float(self.model.predict_proba(processed_data)[0][1])
        prediction = int(probability >= self.threshold)

        logger.info(f"تم التوقع بنجاح: is_late={prediction}, prob={probability:.4f}")

        return {
            "is_late": prediction,
            "late_probability": round(probability, 4),
        }
