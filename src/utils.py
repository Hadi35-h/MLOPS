import logging
import os
import yaml


# إعداد الـ Logger العام للمشروع
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger("olist_mlops")


def setup_logger(config_path="config/config.yaml", name="MLOpsApp") -> logging.Logger:
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    log_file = config.get("logging", {}).get("log_file", "artifacts/app.log")
    os.makedirs(os.path.dirname(log_file), exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

        # إضافة StreamHandler للطباعة على الـ Console أيضاً
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger
