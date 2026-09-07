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

    def predict(self, df: pd.DataFrame):
        # 1. تحويل البيانات باستخدام المحول المحفوظ مسبقاً
        X_processed = self.preprocessor.transform(df)

        # 2. التوقع وحساب الاحتمالية
        probabilities = self.model.predict_proba(X_processed)[:, 1]
        predictions = (probabilities >= self.threshold).astype(int)

        return predictions[0], float(probabilities[0])
