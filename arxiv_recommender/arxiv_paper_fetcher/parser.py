import xml.etree.ElementTree as ET
from typing import Optional, Dict, List


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
        
        title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
        abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
        
        return {"title": title, "abstract": abstract}
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
            title = entry.find("{http://www.w3.org/2005/Atom}title").text.strip()
            abstract = entry.find("{http://www.w3.org/2005/Atom}summary").text.strip()
            papers.append({"title": title, "abstract": abstract})
    except ET.ParseError:
        pass
    
    return papers
