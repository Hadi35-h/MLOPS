import joblib
import mlflow
import yaml


def log_model_to_mlflow(config_path: str = "config/config.yaml"):
    # 1. قراءة التهيئة
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    # 2. تحديد اسم التجربة
    mlflow.set_experiment("Olist_Inference_Pipeline")

    with mlflow.start_run(run_name="Inference_Artifacts_Logging"):
        # تسجيل المعاملات
        mlflow.log_params(config["model_params"])
        mlflow.log_param("model_name", config["model_info"]["name"])
        mlflow.log_param("version", config["model_info"]["version"])

        # تسجيل الـ Artifacts (النموذج والـ Preprocessor والملف التكويني)
        mlflow.log_artifact(config["paths"]["model_path"], artifact_path="models")
        mlflow.log_artifact(
            config["paths"]["preprocessor_path"], artifact_path="preprocessors"
        )
        mlflow.log_artifact(config_path, artifact_path="config")

        print("تم تسجيل النموذج والملفات في MLflow بنجاح!")


if __name__ == "__main__":
    log_model_to_mlflow()
