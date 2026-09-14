import logging
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

logger = logging.getLogger("FastAPIApp")


class Predictor:

    def __init__(
        self,
        model_path: str = None,
        preprocessor_path: str = None,
    ):
        base_dir = Path(__file__).resolve().parent.parent

        self.model_path = (
            Path(model_path) if model_path else base_dir / "models" / "model.pkl"
        )
        self.preprocessor_path = (
            Path(preprocessor_path)
            if preprocessor_path
            else base_dir / "models" / "preprocessor.pkl"
        )

        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"ملف النموذج غير موجود: {self.model_path}")

        if self.preprocessor_path.exists():
            try:
                self.preprocessor = joblib.load(self.preprocessor_path)
            except Exception:
                self.preprocessor = None
        else:
            self.preprocessor = None

    def predict(self, data) -> dict:
        # 1. تحويل المدخلات إلى DataFrame
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        elif isinstance(data, list):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data.copy()
        else:
            raise ValueError(f"نوع البيانات غير مدعوم: {type(data)}")

        # 2. تطبيق المعالجة المسبقة إن وجدت
        if self.preprocessor is not None:
            features = self.preprocessor.transform(df)
        else:
            df_processed = df.copy()

            # تحويل التواريخ
            date_cols = ["order_approved_at", "order_delivered_carrier_date"]
            if all(col in df_processed.columns for col in date_cols):
                approved = pd.to_datetime(df_processed["order_approved_at"])
                carrier = pd.to_datetime(df_processed["order_delivered_carrier_date"])

                df_processed["approval_to_carrier_hours"] = (
                    carrier - approved
                ).dt.total_seconds() / 3600.0
                df_processed["order_approved_hour"] = approved.dt.hour
                df_processed["order_approved_dayofweek"] = approved.dt.dayofweek
                df_processed = df_processed.drop(columns=date_cols)

            # تحويل البيانات إلى أرقام
            df_processed = pd.get_dummies(df_processed, drop_first=True)
            df_processed = df_processed.astype(float)

            # مطابقة الخصائص وتجهيز مصفوفة NumPy
            if hasattr(self.model, "n_features_in_"):
                expected_n = self.model.n_features_in_
                current_n = df_processed.shape[1]
                if current_n < expected_n:
                    for i in range(current_n, expected_n):
                        df_processed[f"feature_{i}"] = 0.0
                elif current_n > expected_n:
                    df_processed = df_processed.iloc[:, :expected_n]

            # تحويل إلى NumPy Array لإلغاء تحذير Feature Names
            features = df_processed.to_numpy()

        # 3. التوقع وتحويل الناتج لـ Python Native Int
        prediction = self.model.predict(features)
        raw_val = prediction[0]
        pred_value = int(raw_val.item()) if hasattr(raw_val, "item") else int(raw_val)

        return {"prediction": pred_value}
