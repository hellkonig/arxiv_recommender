"""Fetcher module for retrieving papers from arXiv API."""

import requests
from typing import Optional

from arxiv_recommender.schemas import Paper
from arxiv_recommender.utils.retry import retry_with_backoff
from arxiv_recommender.utils.logging import get_logger
from .parser import parse_paper_info, parse_papers
from .utils import format_arxiv_query


class ArxivFetcher:
    """A class to fetch paper information from the arXiv API."""

    def __init__(self, max_results: int = 2000, timeout: int = 10):
        """Initializes the fetcher with configurable parameters.

        Args:
            max_results: Maximum results per API call (default: 2000).
            timeout: Timeout for API requests in seconds (default: 10).
        """
        self.base_url = "http://export.arxiv.org/api/query?"
        self.namespace = {"arxiv": "http://www.w3.org/2005/Atom"}
        self.max_results = max_results
        self.timeout = timeout
        self._logger = get_logger(__name__)

    @retry_with_backoff(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
    def get_paper_by_id(self, paper_id: str) -> Optional[Paper]:
        """Fetches a single paper's title and abstract using its arXiv ID.

        Args:
            paper_id: The unique arXiv paper identifier.

        Returns:
            A Paper object containing the paper's title and abstract, or None if not found.

        Raises:
            requests.RequestException: If the API request fails.
        """
        url = "%sid_list=%s" % (self.base_url, paper_id)
        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()

        return parse_paper_info(response.text)

    @retry_with_backoff(max_retries=3, initial_delay=1.0, backoff_factor=2.0)
    def get_daily_papers(
        self, date: Optional[str] = None, category: Optional[str] = None
    ) -> list[Paper]:
        """Fetches papers submitted to arXiv in the last 24 hours or on a specific date.

        Args:
            date: The date for which to fetch papers in YYYYMMDD format.
                If None, fetches papers from the last 24 hours.
            category: The arXiv category (e.g., 'cs.LG' for Machine Learning).

        Returns:
            A list of Paper objects, each containing title and abstract.

        Raises:
            requests.RequestException: If the API request fails.
        """
        query = format_arxiv_query(date, category, max_results=self.max_results)
        self._logger.info("Fetching daily papers with query: %s", query)
        url = "%s%s&max_results=%s" % (self.base_url, query, self.max_results)

        response = requests.get(url, timeout=self.timeout)
        response.raise_for_status()

        papers = parse_papers(response.text)
        self._logger.info(
            "Retrieved %d new papers%s",
            len(papers),
            (" in category %s" % category) if category else "",
        )
        return papers
