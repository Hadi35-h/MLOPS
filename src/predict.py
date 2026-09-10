import logging
import joblib
import pandas as pd
from pathlib import Path

logger = logging.getLogger("FastAPIApp")


class Predictor:
    def __init__(
        self,
        model_path: str = "models/model.pkl",
        preprocessor_path: str = "models/preprocessor.pkl",
    ):
        self.model_path = Path(model_path)
        self.preprocessor_path = Path(preprocessor_path)

        if self.model_path.exists():
            self.model = joblib.load(self.model_path)
        else:
            raise FileNotFoundError(f"ملف النموذج غير موجود: {self.model_path}")

        if self.preprocessor_path.exists():
            try:
                self.preprocessor = joblib.load(self.preprocessor_path)
                logger.info("تم تحميل الـ Preprocessor بنجاح.")
            except Exception as e:
                logger.warning(f"تعذر تحميل الـ Preprocessor: {e}")
                self.preprocessor = None
        else:
            self.preprocessor = None

    def predict(self, data: dict) -> dict:
        try:
            df = pd.DataFrame([data])

            # 1. تطبيق المعالج الأساسي إذا كان موجوداً
            if self.preprocessor is not None:
                features = self.preprocessor.transform(df)
            else:
                # 2. هندسة الخصائص المقابلة لما تم تدريبه في train.py
                df_processed = df.copy()

                # تحويل التواريخ وحساب الفروقات الزمنية
                if (
                    "order_approved_at" in df_processed.columns
                    and "order_delivered_carrier_date" in df_processed.columns
                ):
                    approved = pd.to_datetime(df_processed["order_approved_at"])
                    carrier = pd.to_datetime(
                        df_processed["order_delivered_carrier_date"]
                    )

                    # إنشاء خصائص زمنية تفصيلية لتكتمل الـ 10 الخصائص المطلوبة
                    df_processed["approval_to_carrier_hours"] = (
                        carrier - approved
                    ).dt.total_seconds() / 3600.0
                    df_processed["order_approved_hour"] = approved.dt.hour
                    df_processed["order_approved_dayofweek"] = approved.dt.dayofweek

                    # حذف التواريخ الأصلية بعد استخراج الخصائص منها
                    df_processed = df_processed.drop(
                        columns=["order_approved_at", "order_delivered_carrier_date"]
                    )

                # تحويل المتغيرات النصية إلى إشارات رقمية
                for col in df_processed.select_dtypes(include=["object"]).columns:
                    df_processed[col] = df_processed[col].astype("category").cat.codes

                features = df_processed

            # 3. التأكد من تطابق الخصائص مع ما يتوقعه الموديل
            if hasattr(self.model, "feature_names_in_"):
                # إعادة ترتيب وإكمال أي أعمدة ناقصة تلقائياً بـ 0
                features = features.reindex(
                    columns=self.model.feature_names_in_, fill_value=0
                )

            # 4. إجراء التوقع
            prediction = self.model.predict(features)
            return {"prediction": int(prediction[0])}

        except Exception as e:
            logger.error(f"فشلت عملية التوقع: {e}")
            raise e
