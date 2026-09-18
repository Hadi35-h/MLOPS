import pandas as pd
import numpy as np
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
from evidently.ui.workspace import RemoteWorkspace
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

    # Create the report
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_data, current_data=current_data)

    # Save a local copy
    report.save_html("data_drift_report.html")

    # Send report to Evidently UI via Docker internal network
    try:
        ws = RemoteWorkspace("http://evidently:8080")
        projects = ws.search_project("Olist Monitoring")
        project = projects[0] if projects else ws.create_project("Olist Monitoring")

        ws.add_report(project.id, report)
        logger.info("Report successfully pushed to Evidently UI!")
    except Exception as e:
        logger.warning(f"Could not connect to Evidently UI server: {e}")


if __name__ == "__main__":
    generate_report()
