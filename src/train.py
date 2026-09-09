import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.utils import logger


def train():
    # إعداد تتبع MLflow مع السيرفر المحلي
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("Olist_Churn_Prediction")

    logger.info("Starting model training pipeline...")

    # إنشاء بيانات وهمية متوافقة لغرض الاختبار والتسجيل
    X, y = make_classification(n_samples=1000, n_features=10, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        n_estimators = 100
        max_depth = 5

        clf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, random_state=42
        )
        clf.fit(X_train, y_train)

        predictions = clf.predict(X_test)
        acc = accuracy_score(y_test, predictions)

        # تسجيل المعلمات والمقاييس في MLflow
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(clf, "model")

        logger.info(f"Model trained successfully. Accuracy: {acc:.4f}")


if __name__ == "__main__":
    train()
