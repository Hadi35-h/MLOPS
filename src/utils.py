import logging
from pathlib import Path
import yaml


def load_config(config_path: str = "config/config.yaml") -> dict:
    """قراءة ملف الإعدادات YAML"""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"ملف الإعدادات غير موجود في المسار: {config_path}")

    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logger(name: str = "mlops_logger") -> logging.Logger:
    """إعداد سجل النظام Logging للإنتاج"""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger
