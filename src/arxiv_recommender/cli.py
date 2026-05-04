import argparse
import os

from arxiv_recommender.schemas import AppConfig
from arxiv_recommender.utils import MetricsCollector, setup_logging
from arxiv_recommender.utils.json_handler import load_json
from arxiv_recommender.utils.model_loader import load_vectorization_model
from arxiv_recommender.utils.paper_loader import load_favorite_papers
from arxiv_recommender.utils.user_input import get_favorite_papers_from_user
from arxiv_recommender.utils.logging import get_logger
from arxiv_recommender.recommendation.recommendation import Recommender
from arxiv_recommender.arxiv_paper_fetcher.fetcher import ArxivFetcher


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
        raise FileNotFoundError("Configuration file not found: %s", config_path)
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
    logger = get_logger(__name__)

    metrics = MetricsCollector()

    fetcher = ArxivFetcher()
    favorite_papers = load_favorite_papers(config.favorite_papers_path)
    if not favorite_papers:
        logger.info("No favorite papers provided. Prompting user input...")
        favorite_papers = get_favorite_papers_from_user(config.favorite_papers_path, fetcher)

    vectorizer = load_vectorization_model(
        module_name=config.vectorizer.module_name,
        class_name=config.vectorizer.class_name,
        model_name=config.vectorizer.model_name,
        cache_size=config.vectorizer.cache_size,
    )
    recommender = Recommender(vectorizer, favorite_papers, metrics)

    daily_papers = fetcher.get_daily_papers(date=args.date_of_pulling_papers)
    recommended_papers = recommender.recommend_by_papers(daily_papers, top_k=config.top_k)

    logger.info("Top recommended papers:")
    for i, paper in enumerate(recommended_papers, 1):
        logger.info("%d. %s (%s)", i, paper["title"], paper["abstract"])

    if args.stats or config.log_level == "DEBUG":
        summary = metrics.get_summary()
        cache_stats = vectorizer.get_cache_stats()
        summary["cache"] = cache_stats
        logger.info("Metrics summary: %s", summary)


if __name__ == "__main__":
    main()
