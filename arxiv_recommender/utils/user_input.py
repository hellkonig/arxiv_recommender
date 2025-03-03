import logging
from typing import List

def get_favorite_papers_from_user() -> List[str]:
    """
    Prompts the user to enter arXiv IDs for favorite papers.

    Returns:
        List[str]: A list of arXiv paper IDs.
    """
    logging.info("No favorite papers provided. Please enter arXiv IDs manually.")
    papers = []
    while True:
        paper_id = input("Enter an arXiv paper ID (or press Enter to finish): ").strip()
        if not paper_id:
            break
        papers.append(paper_id)

    if not papers:
        raise ValueError("At least one favorite paper ID must be provided.")

    return papers
