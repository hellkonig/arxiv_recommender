import logging
from typing import List, Dict

from arxiv_paper_fetcher.fetcher import ArxivFetcher
from utils.json_handler import save_json

def get_favorite_papers_from_user(output_file: str) -> List[Dict[str, str]]:
    """
    Prompts the user to enter arXiv IDs, fetches metadata, and saves it.

    Args:
        output_file (str): Path to save the favorite papers' metadata.

    Returns:
        List[Dict[str, str]]: List of dictionaries containing paper metadata.
    """
    logging.info("No favorite papers provided. Enter arXiv IDs manually.")

    fetcher = ArxivFetcher()
    papers = []

    while True:
        paper_id = input(
            "Enter an arXiv paper ID (or press Enter to finish): "
        ).strip()
        if not paper_id:
            break

        try:
            paper_metadata = fetcher.get_paper_by_id(paper_id)
            if paper_metadata:
                papers.append(paper_metadata)
                logging.info(f"Added: {paper_metadata['title']}")
            else:
                logging.warning(f"Paper ID {paper_id} not found.")
        except Exception as e:
            logging.error(f"Error fetching {paper_id}: {e}")

    if not papers:
        raise ValueError("At least one valid arXiv paper is required.")

    save_json(output_file, papers)
    return papers
