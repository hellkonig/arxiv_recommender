import os

from arxiv_recommender.schemas import AppConfig
from arxiv_recommender.utils.json_handler import load_json


def load_config(config_path: str) -> AppConfig:
    """Load and validate an application configuration file.

    Args:
        config_path: Path to the configuration JSON file.

    Returns:
        Parsed application configuration.

    Raises:
        FileNotFoundError: If the configuration file is missing.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config_data = load_json(config_path)
    return AppConfig.model_validate(config_data)
