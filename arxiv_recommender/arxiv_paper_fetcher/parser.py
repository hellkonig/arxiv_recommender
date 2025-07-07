import xml.etree.ElementTree as ET
from typing import Optional, Dict, List

from arxiv_recommender.arxiv_paper_fetcher.utils import remove_control_characters


def extract_metadata(entry: ET.Element) -> Dict[str, str]:
    """
    Extracts paper's meta data from a single XML entry.

    Args:
        entry (ET.Element): An XML element representing a paper entry.

    Returns:
        Dict[str, str]: A dictionary containing 'title' and 'abstract'.
    """
    title = remove_control_characters(
        entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
    )
    abstract = remove_control_characters(
        entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
    )

    if title is None or abstract is None:
        raise ValueError("Title or abstract is missing in the entry.")
    if not title or not abstract:
        raise ValueError("Title or abstract is empty in the entry.")
    if not isinstance(title, str) or not isinstance(abstract, str):
        raise TypeError("Title or abstract is not a string in the entry.")

    return {"title": title, "abstract": abstract}

def parse_paper_info(xml_data: str) -> Optional[Dict[str, str]]:
    """
    Parses a single paper's information (title and abstract) from the arXiv API XML response.

    Args:
        xml_data (str): The XML response from arXiv API.

    Returns:
        Optional[Dict[str, str]]: A dictionary containing 'title' and 'abstract' if successful, else None.
    """
    try:
        root = ET.fromstring(xml_data)
        entry = root.find("{http://www.w3.org/2005/Atom}entry")
        if entry is None:
            return None
        return extract_metadata(entry)
    except ET.ParseError:
        return None

def parse_papers(xml_data: str) -> List[Dict[str, str]]:
    """
    Parses multiple papers' information from the arXiv API XML response.

    Args:
        xml_data (str): The XML response from arXiv API.

    Returns:
        List[Dict[str, str]]: A list of dictionaries, each containing 'title' and 'abstract'.
    """
    papers = []
    try:
        root = ET.fromstring(xml_data)
        for entry in root.findall("{http://www.w3.org/2005/Atom}entry"):
            if entry is None:
                continue
            papers.append(extract_metadata(entry))
    except ET.ParseError:
        pass
    
    return papers
