from typing import Dict, Tuple
import pandas as pd


def validate_input_data(data: Dict) -> Tuple[bool, str]:
    """فحص البيانات المدخلة للتأكد من خلوها من القيم الخاطئة أو الناقصة."""
    df = pd.DataFrame([data])

    # 1. التحقق من الأعمدة الحسابية ألا تكون بالسالب
    numeric_checks = [
        col
        for col in df.columns
        if "price" in col or "freight" in col or "score" in col
    ]
    for col in numeric_checks:
        if (df[col] < 0).any():
            return False, f"Invalid negative value in field: {col}"

    # 2. التحقق من القيم الناقصة
    if df.isnull().any().any():
        return False, "Missing values found in input data"

    return True, "Validation successful"
