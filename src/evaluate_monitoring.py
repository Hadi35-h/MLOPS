import json
import sys
from pathlib import Path
import pandas as pd

# Ensure UTF-8 output on Windows consoles (emoji support)
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8")

# Correct import for modern Evidently releases
try:
    from evidently.metric_preset import DataDriftPreset
    from evidently.report import Report
except ImportError:
    print("❌ evidently is not installed. Run: pip install 'evidently<0.5.0'")
    DataDriftPreset = None
    Report = None

BASE_DIR = Path(__file__).resolve().parent.parent
LOGS_PATH = BASE_DIR / "logs" / "prediction_logs.jsonl"
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def load_prediction_logs(log_file: Path) -> pd.DataFrame:
    """Load prediction logs and convert them to a Pandas DataFrame"""
    if not log_file.exists():
        print(f"❌ No log file found at: {log_file}")
        return pd.DataFrame()

    logs = []
    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                logs.append(json.loads(line))

    if not logs:
        print("⚠️ Log file is empty!")
        return pd.DataFrame()

    records = []
    for entry in logs:
        row = entry["input"].copy()
        row["prediction"] = entry["output"].get("prediction")
        row["probability"] = entry["output"].get("probability")
        row["timestamp"] = entry["timestamp"]
        records.append(row)

    return pd.DataFrame(records)


def evaluate_predictions():
    df_current = load_prediction_logs(LOGS_PATH)

    if df_current.empty:
        print("⚠️ No current prediction logs to evaluate.")
        return

    print(f"📊 Total predictions logged: {len(df_current)}")

    # Use the raw dataset as reference for comparison
    ref_data_path = BASE_DIR / "data" / "raw" / "olist_order_items_dataset.csv"

    if ref_data_path.exists():
        df_reference = pd.read_csv(ref_data_path)

        # Match common columns between reference and current data
        common_cols = [col for col in df_reference.columns if col in df_current.columns]

        if not common_cols:
            print("❌ No common columns between reference data and current logs.")
            return

        # Create the report using DataDriftPreset
        if Report is None or DataDriftPreset is None:
            print("❌ Cannot generate report: evidently not installed.")
            return
        report = Report(metrics=[DataDriftPreset()])
        report.run(
            reference_data=df_reference[common_cols],
            current_data=df_current[common_cols],
        )

        output_html = REPORTS_DIR / "data_drift_report.html"
        report.save_html(str(output_html))
        print(f"✅ Monitoring report saved successfully at: {output_html}")
    else:
        print(f"❌ Reference data file not found at: {ref_data_path}")


if __name__ == "__main__":
    evaluate_predictions()
