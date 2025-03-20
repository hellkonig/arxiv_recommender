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

def load_favorite_papers(favorite_papers_path: str) -> List[Dict[str, str]]:
    """
    Loads or prompts for favorite papers.

    Args:
        favorite_papers_path (str): Path to the favorite papers JSON file.

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
            favorite_papers_path
        )

    logging.info(
        f"Successfully loaded {len(favorite_papers_metadata)} favorite papers."
    )
    return favorite_papers_metadata


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
        favorite_papers_path = config.get(
            "favorite_papers_path",
            "data/favorite_papers.json"
        )
        vectorizer_name = config.get(
            "vectorizer",
            {
                "module": "distil_bert",
                "class": "DistilBERTEmbedding"
            }
        )
        top_k = config.get("top_k", 10)

        favorite_papers_metadata = load_favorite_papers(favorite_papers_path)
        vectorizer = load_vectorization_model(
            module_name=vectorizer_name["module"],
            class_name=vectorizer_name["class"]
        )
        recommender = Recommender(vectorizer, favorite_papers_metadata)

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
