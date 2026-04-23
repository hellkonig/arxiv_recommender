from typing import List, Dict
import os
import logging

from arxiv_recommender.utils.json_handler import load_json, save_json


def load_favorite_papers(
    favorite_papers_path: str,
) -> list[dict[str, str]]:
    """
    Loads favorite papers.

    Args:
        favorite_papers_path (str): Path to the favorite papers JSON file.

    Returns:
        list[dict[str, str]]: List of dictionaries containing paper metadata.
    """
    logging.info(f"Using favorite papers file: {favorite_papers_path}")

    if not os.path.exists(favorite_papers_path):
        logging.info(f"Creating empty favorite papers file: {favorite_papers_path}")
        dir_name = os.path.dirname(favorite_papers_path)
        if dir_name:
            # Ensure the path including directory before creating the file
            os.makedirs(dir_name, exist_ok=True)
        save_json(favorite_papers_path, [])
        return []

    favorite_papers_metadata: list[dict[str, str]] = load_json(favorite_papers_path)

    logging.info(f"Successfully loaded {len(favorite_papers_metadata)} favorite papers.")
    return favorite_papers_metadata
