import logging
import yaml


def setup_logger(config_path: str = "config/config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(config["paths"]["log_file"], encoding="utf-8"),
            logging.StreamHandler(),
        ],
    )
    return logging.getLogger("MLOpsPipeline")


logger = setup_logger()
