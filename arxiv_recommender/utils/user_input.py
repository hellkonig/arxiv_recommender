import logging

from arxiv_recommender.schemas import Paper
from arxiv_recommender.utils.json_handler import save_json
from arxiv_recommender.arxiv_paper_fetcher.fetcher import ArxivFetcher


def get_favorite_papers_from_user(output_file: str, fetcher: ArxivFetcher) -> list[Paper]:
    """
    Prompts the user to enter arXiv IDs, fetches metadata, and saves it.

    Args:
        output_file (str): Path to save the favorite papers' metadata.
        fetcher (ArxivFetcher): Instance responsible for fetching metadata.

    Returns:
        list[Paper]: List of Paper objects containing paper metadata.
    """
    logging.info("No favorite papers provided. Enter arXiv IDs manually.")

    papers = []

    while True:
        paper_id = input("Enter an arXiv paper ID (or press Enter to finish): ").strip()
        if not paper_id:
            break

        try:
            paper = fetcher.get_paper_by_id(paper_id)
            if paper:
                papers.append(paper)
                logging.info(f"Added: {paper.title}")
            else:
                logging.warning(f"Paper ID {paper_id} not found.")
        except Exception as e:
            logging.error(f"Error fetching {paper_id}: {e}")

    if not papers:
        raise ValueError("At least one valid arXiv paper is required.")

    save_json(output_file, [paper.model_dump() for paper in papers])

    logging.info(f"Loaded {len(papers)} favorite papers and saved them to {output_file}.")
    return papers