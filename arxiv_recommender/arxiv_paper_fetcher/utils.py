from datetime import datetime, timedelta
from typing import Optional


def format_arxiv_query(
        date: Optional[str] = None,
        category: Optional[str] = None,
        max_results: int = 50
    ) -> str:
    """
    Formats the search query for arXiv API to fetch papers from the last 24 hours.

    Args:
        date (Optional[str]): The date for which to fetch papers in YYYYMMDD format.
            If provided, it should be in the format 'YYYYMMDD'.
            If None, the function will fetch papers from the last 24 hours.
        category (Optional[str]): The arXiv category (e.g., 'cs.LG' for Machine Learning).
        max_results (int): The maximum number of results to fetch (default is 50).

    Returns:
        str: The formatted query string.
    """
    if date:
        # If a specific date is provided, use it to format the query
        try:
            date_obj = datetime.strptime(date, '%Y%m%d')
        except ValueError:
            raise ValueError("Date must be in YYYYMMDD format.")
        date_str = date_obj.strftime('%Y%m%d')
    else:
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

    >>> text = "Hello\nWorld!\tThis is a test.\r\n"
    >>> remove_control_characters(text)
    'Hello World! This is a test.'

    Args:
        text (str): The input string.

    Returns:
        str: The cleaned string with control characters removed.
    """
    text_words = []
    for c in text:
        if not c.isprintable() or c == ' ':
            if text_words and text_words[-1] != ' ':
                text_words.append(' ')
        else:
            text_words.append(c)
    text = ''.join(text_words)
    return text.strip()
