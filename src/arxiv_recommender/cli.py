import argparse
import logging
import os

from arxiv_recommender.recommendation import RecommendationPipeline
from arxiv_recommender.schemas import AppConfig
from arxiv_recommender.utils import setup_logging
from arxiv_recommender.utils.json_handler import load_json


def load_config(config_path: str) -> AppConfig:
    """Loads the configuration file.

    Args:
        config_path: Path to the configuration JSON file.

    Returns:
        Parsed configuration object.

    Raises:
        FileNotFoundError: If the configuration file is missing.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    config_data = load_json(config_path)
    return AppConfig.model_validate(config_data)


def main() -> None:
    parser = argparse.ArgumentParser(description="arXiv Paper Recommender System")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file.",
    )
    parser.add_argument(
        "--date_of_pulling_papers",
        type=str,
        default=None,
        help="Date of pulling papers in YYYYMMDD format. If not provided, defaults to today.",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=None,
        help="Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).",
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Print metrics summary at the end of execution.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    log_level = args.log_level or config.log_level
    setup_logging(level=log_level, json_format=True)
    logger = logging.getLogger(__name__)

    pipeline = RecommendationPipeline(config=config)
    result = pipeline.run(date_of_pulling_papers=args.date_of_pulling_papers)

    logger.info("Top recommended papers:")
    for i, paper in enumerate(result.recommendations, 1):
        logger.info("%d. %s (%s)", i, paper["title"], paper["abstract"])

    if args.stats or log_level == "DEBUG":
        logger.info("Metrics summary: %s", result.metrics_summary)


if __name__ == "__main__":
    main()
