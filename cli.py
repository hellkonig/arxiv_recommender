import os
import argparse
import logging
from typing import Dict, Any

from arxiv_recommender.utils.json_handler import load_json, save_json
from arxiv_recommender.utils.model_loader import load_vectorization_model
from arxiv_recommender.utils.user_input import get_favorite_papers_from_user
from arxiv_recommender.arxiv_paper_fetcher.fetcher import ArxivFetcher
from arxiv_recommender.recommendation.recommendation import (
    Recommender,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Constants
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
CONFIG_PATH = os.path.join(DATA_DIR, "config.json")


def ensure_data_directory() -> None:
    """Ensures the 'data' directory exists before saving any files."""
    os.makedirs(DATA_DIR, exist_ok=True)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Loads the configuration file.

    Args:
        config_path (str): Path to the configuration JSON file.

    Returns:
        Dict[str, Any]: Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the configuration file is missing.
    """
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    return load_json(config_path)


def load_favorite_papers(config: Dict[str, Any]) -> str:
    """
    Loads or prompts for favorite papers.

    Args:
        config (Dict[str, Any]): Parsed configuration dictionary.

    Returns:
        str: Path to the favorite papers JSON file.

    Raises:
        ValueError: If no favorite papers file is provided.
    """
    favorite_papers_path = config.get("favorite_papers_path")

    if not favorite_papers_path:
        favorite_papers_path = os.path.join(DATA_DIR, "favorite_papers.json")
        logging.info(f"Defaulting to {favorite_papers_path}")

    if not os.path.exists(favorite_papers_path):
        logging.info(
            f"Creating empty favorite papers file: {favorite_papers_path}"
        )
        os.makedirs(os.path.dirname(favorite_papers_path), exist_ok=True)
        save_json(favorite_papers_path, [])

    favorite_papers_metadata = load_json(favorite_papers_path)
    if not favorite_papers_metadata:
        logging.info("No favorite papers provided. Prompting user input...")
        get_favorite_papers_from_user(favorite_papers_path)

    return favorite_papers_path


def main():
    """Main function to run the CLI application."""
    ensure_data_directory()

    parser = argparse.ArgumentParser(
        description="arXiv Paper Recommender System"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=CONFIG_PATH,
        help="Path to configuration JSON file.",
    )

    args = parser.parse_args()

    try:
        config = load_config(args.config)

        # Read values from config
        favorite_papers_path = load_favorite_papers(config)
        vectorizer_name = config.get(
            "vectorizer",
            "text_vectorization.distill_bert.DistilBERTEmbedding",
        )
        top_k = config.get("top_k", 10)

        vectorizer = load_vectorization_model(vectorizer_name)
        fetcher = ArxivFetcher()
        recommender = Recommender(fetcher, vectorizer)

        recommended_papers = recommender.recommend_by_papers(
            favorite_papers_path, top_k=top_k
        )

        logging.info("Top recommended papers:")
        for i, paper in enumerate(recommended_papers, 1):
            logging.info(f"{i}. {paper['title']} ({paper['arxiv_id']})")

    except Exception as e:
        logging.error(f"Error: {e}")
        exit(1)


if __name__ == "__main__":
    main()
