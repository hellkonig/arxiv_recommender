import requests
import logging
from typing import Optional, List, Dict

from .parser import parse_paper_info, parse_papers
from .utils import format_arxiv_query


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

class ArxivFetcher:
    """A class to fetch paper information from the arXiv API."""

    def __init__(self, max_results: int = 2000, timeout: int = 10):
        """
        Initializes the fetcher with configurable parameters.

        Args:
            max_results (int): Maximum results per API call (default: 2000).
            timeout (int): Timeout for API requests in seconds (default: 10).
        """
        self.base_url = "http://export.arxiv.org/api/query?"
        self.namespace = {"arxiv": "http://www.w3.org/2005/Atom"}
        self.max_results = max_results
        self.timeout = timeout

    def get_paper_by_id(self, paper_id: str) -> Optional[Dict[str, str]]:
        """
        Fetches a single paper's title and abstract using its arXiv ID.

        Args:
            paper_id (str): The unique arXiv paper identifier.

        Returns:
            dict: A dictionary containing the paper's title and abstract, or None if not found.

        Raises:
            requests.RequestException: If the API request fails.
        """
        url = f"{self.base_url}id_list={paper_id}"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Failed to fetch paper {paper_id}: {e}")
            raise

        return parse_paper_info(response.text)

    def get_daily_papers(self, date: Optional[str] = None, category: Optional[str] = None) -> List[Dict[str, str]]:
        """
        Fetches papers submitted to arXiv in the last 24 hours or on a specific date.

        Args:
            date (Optional[str]): The date for which to fetch papers in YYYYMMDD format.
                If provided, it should be in the format 'YYYYMMDD'.
                If None, the function will fetch papers from the last 24 hours.
            category (Optional[str]): The arXiv category (e.g., 'cs.LG' for Machine Learning).

        Returns:
            List[Dict[str, str]]: A list of dictionaries, each containing title and abstract.

        Raises:
            requests.RequestException: If the API request fails.
        """
        query = format_arxiv_query(date, category, max_results=self.max_results)
        logging.info(f"Fetching daily papers with query: {query}")
        url = f"{self.base_url}{query}&max_results={self.max_results}"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
        except requests.RequestException as e:
            logging.error(f"Failed to fetch daily papers: {e}")
            raise

        papers = parse_papers(response.text)
        logging.info(f"Retrieved {len(papers)} new papers" + (f" in category {category}" if category else ""))
        return papers
