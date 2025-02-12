from datetime import datetime, timedelta
from typing import Optional


def format_arxiv_query(category: Optional[str] = None) -> str:
    """
    Formats the search query for arXiv API to fetch papers from the last 24 hours.

    Args:
        category (Optional[str]): The arXiv category (e.g., 'cs.LG' for Machine Learning).

    Returns:
        str: The formatted query string.
    """
    yesterday = datetime.utcnow() - timedelta(days=1)
    date_str = yesterday.strftime('%Y%m%d')
    query = f'submittedDate:[{date_str}0000 TO {date_str}2359]'
    
    if category:
        query = f'cat:{category} AND {query}'
    
    return query
