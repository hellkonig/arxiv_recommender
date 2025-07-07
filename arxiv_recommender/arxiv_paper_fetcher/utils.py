from datetime import datetime, timedelta
from typing import Optional


def format_arxiv_query(
        category: Optional[str] = None,
        max_results: int = 50
    ) -> str:
    """
    Formats the search query for arXiv API to fetch papers from the last 24 hours.

    Args:
        category (Optional[str]): The arXiv category (e.g., 'cs.LG' for Machine Learning).
        max_results (int): The maximum number of results to fetch (default is 50).

    Returns:
        str: The formatted query string.
    """
    yesterday = datetime.now().astimezone() - timedelta(days=1)
    date_str = yesterday.strftime('%Y%m%d')
    query = f'search_query=submittedDate:[{date_str}0000+TO+{date_str}2359]&max_results={max_results}'
    
    if category:
        query = f'search_query=cat:{category}+AND+submittedDate:[{date_str}0000+TO+{date_str}2359]&max_results={max_results}'
    
    return query

def remove_control_characters(text: str) -> str:
    """
    Removes control characters from a string.
    The control characters are non-printable characters that can cause issues in text processing, e.g., \n, \r, \t, etc.

    Args:
        text (str): The input string.

    Returns:
        str: The cleaned string with control characters removed.
    """
    return ''.join(c for c in text if c.isprintable())
