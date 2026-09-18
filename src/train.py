import os
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib

# Set MLflow tracking URI dynamically based on environment
mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(mlflow_uri)


def train():
    # Set the experiment name
    mlflow.set_experiment("Olist_Churn_Prediction")

    # Generate synthetic data for experimentation
    X = np.random.rand(100, 10)
    y = np.random.randint(0, 2, 100)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        n_estimators = 100
        max_depth = 5

        # Log parameters (Parameters)
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

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

        # Log metrics (Metrics)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("precision", prec)
        mlflow.log_metric("recall", rec)
        mlflow.log_metric("f1_score", f1)

        # Save model locally and register in MLflow
        os.makedirs("models", exist_ok=True)
        joblib.dump(model, "models/model.pkl")
        mlflow.sklearn.log_model(model, "model")

        print(f"Model trained successfully. Accuracy: {acc:.4f}")


if __name__ == "__main__":
    train()
