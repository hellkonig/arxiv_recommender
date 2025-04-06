import os
import argparse
import logging
from typing import Dict, List, Any

from arxiv_recommender.utils.json_handler import load_json, save_json
from arxiv_recommender.utils.model_loader import load_vectorization_model
from arxiv_recommender.utils.user_input import get_favorite_papers_from_user
from arxiv_recommender.recommendation.recommendation import (
    Recommender,
)
from arxiv_recommender.arxiv_paper_fetcher.fetcher import ArxivFetcher


# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


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


def load_favorite_papers(
        favorite_papers_path: str,
        fetcher: ArxivFetcher
    ) -> List[Dict[str, str]]:
    """
    Loads or prompts for favorite papers.

    Args:
        favorite_papers_path (str): Path to the favorite papers JSON file.
        fetcher (ArxivFetcher): Instance responsible for fetching metadata.

    Returns:
        List[Dict[str, str]]: List of favorite papers' metadata.
    """
    logging.info(f"Using favorite papers file: {favorite_papers_path}")

    if not os.path.exists(favorite_papers_path):
        logging.info(
            f"Creating empty favorite papers file: {favorite_papers_path}"
        )
        os.makedirs(os.path.dirname(favorite_papers_path), exist_ok=True)
        save_json(favorite_papers_path, [])

    favorite_papers_metadata: List[Dict[str, str]] = load_json(
        favorite_papers_path
    )

    if not favorite_papers_metadata:
        logging.info("No favorite papers provided. Prompting user input...")
        favorite_papers_metadata = get_favorite_papers_from_user(
            favorite_papers_path,
            fetcher,
        )

    logging.info(
        f"Successfully loaded {len(favorite_papers_metadata)} favorite papers."
    )
    return favorite_papers_metadata


def main():
    parser = argparse.ArgumentParser(
        description="arXiv Paper Recommender System"
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration JSON file.",
    )
    args = parser.parse_args()

    config = load_config(args.config)

    # Read values from config
    favorite_papers_path = config.get(
        "favorite_papers_path",
        "arxiv_recommender/data/favorite_papers.json"
    )
    vectorizer_name = config.get(
        "vectorizer",
        {
            "module": "distil_bert",
            "class": "DistilBERTEmbedding",
            "model": "arxiv_recommender/data/models/distilbert"
        }
    )
    top_k = config.get("top_k", 10)

    fetcher = ArxivFetcher()
    favorite_papers_metadata = load_favorite_papers(
        favorite_papers_path,
        fetcher
    )
    vectorizer = load_vectorization_model(
        module_name=vectorizer_name["module"],
        class_name=vectorizer_name["class"],
        model_name=vectorizer_name["model"],
    )
    recommender = Recommender(vectorizer, favorite_papers_metadata)

    daily_papers = fetcher.get_daily_papers()
    recommended_papers = recommender.recommend_by_papers(
        daily_papers, top_k=top_k
    )

    logging.info("Top recommended papers:")
    for i, paper in enumerate(recommended_papers, 1):
        logging.info(f"{i}. {paper['title']} ({paper['abstract']})")


if __name__ == "__main__":
    main()
