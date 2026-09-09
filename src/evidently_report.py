import pandas as pd
import numpy as np
from evidently import metrics
from src.utils import logger


def generate_report():
    logger.info("Generating Data Drift Report using Evidently...")

    np.random.seed(42)
    reference_data = pd.DataFrame(
        {
            "feature1": np.random.normal(0, 1, 100),
            "feature2": np.random.normal(5, 2, 100),
        }
    )

    current_data = pd.DataFrame(
        {
            "feature1": np.random.normal(0.5, 1, 100),
            "feature2": np.random.normal(5, 2, 100),
        }
    )

    # استدعاء تقرير انحراف البيانات بالطريقة المباشرة
    from evidently.report import Report
    from evidently.metric_preset import DataDriftPreset

    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    report.save_html("data_drift_report.html")
    logger.info(
        "Data Drift Report generated successfully and saved as 'data_drift_report.html'."
    )


if __name__ == "__main__":
    generate_report()
