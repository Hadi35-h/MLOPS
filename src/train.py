import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

try:
    import mlflow
    import mlflow.sklearn

    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    mlflow.set_tracking_uri(mlflow_uri)
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODELS_DIR / "model.pkl"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.pkl"

# Columns expected by the API (OrderInput schema)
NUMERIC_COLS = [
    "total_price",
    "total_freight",
    "total_items",
    "total_payment",
    "max_installments",
]
CATEGORICAL_COLS = ["order_status", "customer_state"]


def create_synthetic_data(n_samples: int = 200) -> pd.DataFrame:
    """Generate synthetic Olist-style order data for training."""
    np.random.seed(42)
    data = {
        "total_price": np.random.uniform(10, 1000, n_samples),
        "total_freight": np.random.uniform(0, 100, n_samples),
        "total_items": np.random.randint(1, 20, n_samples),
        "total_payment": np.random.uniform(20, 1500, n_samples),
        "max_installments": np.random.randint(1, 12, n_samples),
        "order_status": np.random.choice(
            ["delivered", "shipped", "canceled"], n_samples
        ),
        "order_approved_at": pd.date_range(
            "2017-01-01", periods=n_samples, freq="h"
        ).astype(str),
        "order_delivered_carrier_date": pd.date_range(
            "2017-01-02", periods=n_samples, freq="h"
        ).astype(str),
        "customer_state": np.random.choice(["SP", "RJ", "MG", "RS"], n_samples),
    }
    # Synthetic target: 1 if late, 0 otherwise
    df = pd.DataFrame(data)
    df["is_late"] = np.random.randint(0, 2, n_samples)
    return df


def build_preprocessor() -> ColumnTransformer:
    """Create a preprocessing pipeline matching the API input schema."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_COLS),
        ]
    )
    return preprocessor


def train():
    if MLFLOW_AVAILABLE:
        mlflow.set_experiment("Olist_Churn_Prediction")

    # Generate synthetic data
    df = create_synthetic_data(n_samples=200)
    y = df["is_late"]
    X = df.drop(columns=["is_late"])

    # Build and fit the preprocessor
    preprocessor = build_preprocessor()
    X_features = preprocessor.fit_transform(X)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_features, y, test_size=0.2, random_state=42
    )

    # Model parameters
    n_estimators = 100
    max_depth = 5

    # Train the model
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    # Predict and calculate metrics
    predictions = model.predict(X_test)
    acc = accuracy_score(y_test, predictions)
    prec = precision_score(y_test, predictions, zero_division=0)
    rec = recall_score(y_test, predictions, zero_division=0)
    f1 = f1_score(y_test, predictions, zero_division=0)

    # Log to MLflow (optional)
    if MLFLOW_AVAILABLE:
        try:
            with mlflow.start_run():
                mlflow.log_param("n_estimators", n_estimators)
                mlflow.log_param("max_depth", max_depth)
                mlflow.log_metric("accuracy", acc)
                mlflow.log_metric("precision", prec)
                mlflow.log_metric("recall", rec)
                mlflow.log_metric("f1_score", f1)
                mlflow.sklearn.log_model(model, "model")
        except Exception as e:
            print(f"⚠️ MLflow logging failed (training continues): {e}")

    # Save model and preprocessor locally
    joblib.dump(model, str(MODEL_PATH))
    joblib.dump(preprocessor, str(PREPROCESSOR_PATH))

    print(f"Model trained successfully. Accuracy: {acc:.4f}")
    print(f"Model saved to: {MODEL_PATH}")
    print(f"Preprocessor saved to: {PREPROCESSOR_PATH}")


if __name__ == "__main__":
    train()
