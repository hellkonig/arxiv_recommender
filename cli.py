import argparse
import json
import logging
from pathlib import Path

from arxiv_recommender.utils.json_handler import load_json
from arxiv_recommender.utils.user_input import get_favorite_papers_from_user
from arxiv_recommender.utils.model_loader import load_vectorization_model
from arxiv_recommender.recommendation.recommendation import Recommender

logger = logging.getLogger(__name__)


def parse_arguments():
    """
    Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="ArXiv Paper Recommendation System"
    )
    parser.add_argument(
        "--config",
        type=Path,
        required=True,
        help="Path to the JSON configuration file.",
    )
    return parser.parse_args()

def main():
    """
    Command-line interface for the arXiv recommendation system.
    """
    args = parse_arguments()

    # Load configuration
    try:
        with open(args.config, "r", encoding="utf-8") as file:
            config = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logging.error(f"Failed to load config file: {e}")
        return

    favorite_papers_path = config.get("favorite_papers_path")
    filter_field = config.get("filter_field", None)

    #try:
    #    favorite_papers = load_json(favorite_papers_path)
    #    if not favorite_papers:
    #        raise ValueError("Favorite papers file is empty.")
    #except (FileNotFoundError, ValueError):
    #    logger.info("No valid favorite papers found. Requesting user input.")
    #    favorite_papers = get_favorite_papers_from_user(favorite_papers_path)

    #vectorizer = load_vectorization_model()
    #recommender = Recommender(vectorizer)

    #recommended_papers = recommender.recommend_by_papers(
    #    favorite_papers, filter_field=filter_field
    #)

    #for rank, paper in enumerate(recommended_papers, start=1):
    #    print(f"{rank}. {paper['title']} ({paper['id']})")


if __name__ == "__main__":
    main()
