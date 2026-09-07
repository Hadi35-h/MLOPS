import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """تطبيق عمليات التنظيف والمعالجة الأولية على البيانات الخام."""
    df = df.copy()

    # مثال: تحويل التواريخ إذا كانت موجودة
    date_cols = [col for col in df.columns if "date" in col or "timestamp" in col]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # التعامل مع القيم المفقودة الأساسية للأعمدة الرقمية
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # التعامل مع القيم المفقودة للأعمدة النصية
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna("Unknown")

    return df
