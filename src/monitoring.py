import os
import pandas as pd
from evidently.metric_preset import DataDriftPreset
from evidently.report import Report


def generate_drift_report(
    reference_path: str,
    current_path: str,
    output_path: str = "artifacts/drift_report.html",
):
    """إنشاء تقرير انحراف البيانات بين بيانات التدريب والبيانات الحالية."""
    reference_df = pd.read_csv(reference_path)
    current_df = pd.read_csv(current_path)

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    report.save_html(output_path)
    print(f"تم إنشاء تقرير انحراف البيانات بنجاح في: {output_path}")


if __name__ == "__main__":
    # تشغيل تجريبي في حال توفر الملفات
    pass
