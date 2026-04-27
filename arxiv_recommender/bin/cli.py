import os
import argparse
import logging

from arxiv_recommender.schemas import AppConfig
from arxiv_recommender.utils.json_handler import load_json
from arxiv_recommender.utils.model_loader import load_vectorization_model
from arxiv_recommender.utils.paper_loader import load_favorite_papers
from arxiv_recommender.utils.user_input import get_favorite_papers_from_user
from arxiv_recommender.recommendation.recommendation import Recommender
from arxiv_recommender.arxiv_paper_fetcher.fetcher import ArxivFetcher

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_config(config_path: str) -> AppConfig:
    """
    Loads the configuration file.

    Args:
        config_path (str): Path to the configuration JSON file.

    Returns:
        AppConfig: Parsed configuration object.

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
    args = parser.parse_args()

    config = load_config(args.config)

    fetcher = ArxivFetcher()
    favorite_papers = load_favorite_papers(
        config.favorite_papers_path,
    )
    if not favorite_papers:
        logging.info("No favorite papers provided. Prompting user input...")
        favorite_papers = get_favorite_papers_from_user(config.favorite_papers_path, fetcher)

    vectorizer = load_vectorization_model(
        module_name=config.vectorizer.module,
        class_name=config.vectorizer.class_name,
        model_name=config.vectorizer.model,
    )
    recommender = Recommender(vectorizer, favorite_papers)

    daily_papers = fetcher.get_daily_papers(date=args.date_of_pulling_papers)
    recommended_papers = recommender.recommend_by_papers(daily_papers, top_k=config.top_k)

    logging.info("Top recommended papers:")
    for i, paper in enumerate(recommended_papers, 1):
        logging.info(f"{i}. {paper['title']} ({paper['abstract']})")


if __name__ == "__main__":
    main()