import re
from datetime import datetime, timedelta

ARXIV_DATE_PATTERN = re.compile(r"^\d{8}$")
ARXIV_CATEGORY_PATTERN = re.compile(r"^[a-z]+(?:-[a-z]+)*(?:\.[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*)?$")


def validate_arxiv_date(date: str) -> str:
    """Validate an arXiv API date string.

    Args:
        date: Date in YYYYMMDD format.

    Returns:
        The validated date.

    Raises:
        ValueError: If the date is not a real date in YYYYMMDD format.
    """
    if not ARXIV_DATE_PATTERN.fullmatch(date):
        raise ValueError("Date must be in YYYYMMDD format.")
    try:
        datetime.strptime(date, "%Y%m%d")
    except ValueError:
        raise ValueError("Date must be in YYYYMMDD format.") from None
    return date


def validate_arxiv_category(category: str) -> str:
    """Validate an arXiv category string.

    Args:
        category: arXiv category such as 'cs.LG'.

    Returns:
        The validated category.

    Raises:
        ValueError: If the category is not in an arXiv category shape.
    """
    if not ARXIV_CATEGORY_PATTERN.fullmatch(category):
        raise ValueError("Category must be an arXiv category such as 'cs.LG'.")
    return category


def build_arxiv_query_params(
    date: str | None = None, category: str | None = None, max_results: int = 50
) -> dict[str, str | int]:
    """
    Builds search parameters for the arXiv API.

    Args:
        date (Optional[str]): The date for which to fetch papers in YYYYMMDD format.
            If provided, it should be in the format 'YYYYMMDD'.
            If None, the function will fetch papers from the last 24 hours.
        category (Optional[str]): The arXiv category (e.g., 'cs.LG' for Machine Learning).
        max_results (int): The maximum number of results to fetch (default is 50).

    Returns:
        dict[str, str | int]: Query parameters for requests.get(..., params=...).
    """
    if date is not None:
        date_str = validate_arxiv_date(date)
    else:
        yesterday = datetime.now().astimezone() - timedelta(days=1)
        date_str = yesterday.strftime("%Y%m%d")

    submitted_date_query = f"submittedDate:[{date_str}0000 TO {date_str}2359]"
    if category is not None:
        category = validate_arxiv_category(category)
        search_query = f"cat:{category} AND {submitted_date_query}"
    else:
        search_query = submitted_date_query

    return {"search_query": search_query, "max_results": max_results}


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
    text_words: list[str] = []
    for c in text:
        if not c.isprintable() or c == " ":
            if text_words and text_words[-1] != " ":
                text_words.append(" ")
        else:
            text_words.append(c)
    text = "".join(text_words)
    return text.strip()
