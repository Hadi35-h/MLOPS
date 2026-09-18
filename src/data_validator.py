from typing import Dict, Tuple
import pandas as pd


def validate_input_data(data: Dict) -> Tuple[bool, str]:
    """Validate input data to ensure it is free of incorrect or missing values."""
    df = pd.DataFrame([data])

    # 1. Check that numeric columns are not negative
    numeric_checks = [
        col
        for col in df.columns
        if "price" in col or "freight" in col or "score" in col
    ]
    for col in numeric_checks:
        if (df[col] < 0).any():
            return False, f"Invalid negative value in field: {col}"

    # 2. Check for missing values
    if df.isnull().any().any():
        return False, "Missing values found in input data"

    return True, "Validation successful"
