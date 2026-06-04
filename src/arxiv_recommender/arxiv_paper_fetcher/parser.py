import xml.etree.ElementTree as ET
from datetime import datetime
from typing import Optional

from arxiv_recommender.arxiv_paper_fetcher.utils import remove_control_characters
from arxiv_recommender.schemas import Paper

ATOM_NAMESPACES = {"atom": "http://www.w3.org/2005/Atom"}


class ArxivParseError(ValueError):
    """Raised when an arXiv API response cannot be parsed as valid XML."""


def _get_text(entry: ET.Element, field: str) -> str:
    """Return normalized text for an Atom child element."""
    element = entry.find(f"atom:{field}", ATOM_NAMESPACES)
    return (
        remove_control_characters(element.text.strip())
        if element is not None and element.text
        else ""
    )


def _extract_arxiv_id(url: str) -> str:
    """Extract an arXiv identifier from the canonical entry URL."""
    return url.rsplit("/abs/", maxsplit=1)[-1]


def _get_datetime(entry: ET.Element, field: str) -> datetime | None:
    """Return an Atom timestamp as a datetime when present."""
    value = _get_text(entry, field)
    if not value:
        return None
    normalized_value = value.removesuffix("Z") + "+00:00" if value.endswith("Z") else value
    return datetime.fromisoformat(normalized_value)


def extract_metadata(entry: ET.Element) -> Paper:
    """
    Extracts paper's meta data from a single XML entry.

    Args:
        entry (ET.Element): An XML element representing a paper entry.

    Returns:
        Paper: A Paper object containing the available arXiv metadata.
    """
    url = _get_text(entry, "id")
    title = _get_text(entry, "title")
    abstract = _get_text(entry, "summary")
    authors = [
        name
        for author in entry.findall("atom:author", ATOM_NAMESPACES)
        if (name := _get_text(author, "name"))
    ]
    categories = [
        term
        for category in entry.findall("atom:category", ATOM_NAMESPACES)
        if (term := category.get("term"))
    ]

    if not title or not abstract:
        raise ValueError("Title or abstract is empty in the entry.")

    return Paper(
        arxiv_id=_extract_arxiv_id(url) if url else None,
        url=url or None,
        title=title,
        abstract=abstract,
        authors=authors,
        categories=categories,
        published=_get_datetime(entry, "published"),
        updated=_get_datetime(entry, "updated"),
    )


def parse_paper_info(xml_data: str) -> Optional[Paper]:
    """
    Parses a single paper's information (title and abstract) from the arXiv API XML response.

    Args:
        xml_data (str): The XML response from arXiv API.

    Returns:
        Optional[Paper]: A Paper object containing available metadata if successful, else None.
    """
    try:
        root = ET.fromstring(xml_data)
        entry = root.find("atom:entry", ATOM_NAMESPACES)
        if entry is None:
            return None
        return extract_metadata(entry)
    except ET.ParseError as exc:
        raise ArxivParseError("Malformed arXiv XML response.") from exc


def parse_papers(xml_data: str) -> list[Paper]:
    """
    Parses multiple papers' information from the arXiv API XML response.

    Args:
        xml_data (str): The XML response from arXiv API.

    Returns:
        list[Paper]: A list of Paper objects containing available metadata.
    """
    papers = []
    try:
        root = ET.fromstring(xml_data)
        for entry in root.findall("atom:entry", ATOM_NAMESPACES):
            if entry is None:
                continue
            papers.append(extract_metadata(entry))
    except ET.ParseError as exc:
        raise ArxivParseError("Malformed arXiv XML response.") from exc

    return papers
