import pandas as pd


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """Apply cleaning and preprocessing operations to raw data."""
    df = df.copy()

    # Convert date columns if they exist
    date_cols = [col for col in df.columns if "date" in col or "timestamp" in col]
    for col in date_cols:
        df[col] = pd.to_datetime(df[col], errors="coerce")

    # Handle missing values for numeric columns
    num_cols = df.select_dtypes(include=["float64", "int64"]).columns
    for col in num_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

    # Handle missing values for categorical columns
    cat_cols = df.select_dtypes(include=["object", "category"]).columns
    for col in cat_cols:
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna("Unknown")

    return df
