import unittest
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

from arxiv_recommender.arxiv_paper_fetcher.parser import (
    ArxivParseError,
    extract_metadata,
    parse_paper_info,
    parse_papers,
)


class TestParser(unittest.TestCase):
    def setUp(self) -> None:
        """Set up test XML responses"""
        self.sample_entry = """
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <id>http://arxiv.org/abs/1234.56789</id>
                <title>Sample Paper</title>
                <summary>Sample Abstract</summary>
                <author><name>Ada Lovelace</name></author>
                <author><name>Grace Hopper</name></author>
                <category term="cs.AI"/>
                <category term="cs.LG"/>
                <published>2026-05-01T12:00:00Z</published>
                <updated>2026-05-02T12:00:00Z</updated>
            </entry>
        </feed>
        """

        self.sample_feed = """
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <title>Paper 1</title>
                <summary>Abstract 1</summary>
            </entry>
            <entry>
                <title>Paper 2</title>
                <summary>Abstract 2</summary>
            </entry>
        </feed>
        """

    def test_extract_metadata(self) -> None:
        """Test extracting metadata from a single entry"""
        root = ET.fromstring(self.sample_entry)
        entry = root.find("{http://www.w3.org/2005/Atom}entry")
        assert entry is not None
        metadata = extract_metadata(entry)
        self.assertEqual(metadata.arxiv_id, "1234.56789")
        self.assertEqual(metadata.url, "http://arxiv.org/abs/1234.56789")
        self.assertEqual(metadata.title, "Sample Paper")
        self.assertEqual(metadata.abstract, "Sample Abstract")
        self.assertEqual(metadata.authors, ["Ada Lovelace", "Grace Hopper"])
        self.assertEqual(metadata.categories, ["cs.AI", "cs.LG"])
        assert metadata.published is not None
        assert metadata.updated is not None
        self.assertEqual(metadata.published.isoformat(), "2026-05-01T12:00:00+00:00")
        self.assertEqual(metadata.updated.isoformat(), "2026-05-02T12:00:00+00:00")

    def test_parse_paper_info(self) -> None:
        """Test extracting title and abstract from a single entry"""
        paper = parse_paper_info(self.sample_entry)
        assert paper is not None
        self.assertEqual(paper.title, "Sample Paper")
        self.assertEqual(paper.abstract, "Sample Abstract")

    def test_parse_paper_info_parses_utc_z_timestamps(self) -> None:
        """Test UTC timestamps use a Python 3.10-compatible representation."""
        paper = parse_paper_info(self.sample_entry)

        assert paper is not None
        self.assertEqual(paper.published, datetime(2026, 5, 1, 12, tzinfo=timezone.utc))
        self.assertEqual(paper.updated, datetime(2026, 5, 2, 12, tzinfo=timezone.utc))

    def test_parse_paper_info_empty_feed_returns_none(self) -> None:
        """Test parsing a valid feed with no entry returns None."""
        paper = parse_paper_info('<feed xmlns="http://www.w3.org/2005/Atom"></feed>')
        self.assertIsNone(paper)

    def test_parse_paper_info_malformed_xml_raises(self) -> None:
        """Test malformed XML raises an explicit parse error."""
        with self.assertRaises(ArxivParseError):
            parse_paper_info("<feed>")

    def test_parse_papers(self) -> None:
        """Test extracting multiple papers from a feed"""
        papers = parse_papers(self.sample_feed)
        self.assertEqual(len(papers), 2)
        self.assertEqual(papers[0].title, "Paper 1")
        self.assertEqual(papers[1].abstract, "Abstract 2")

    def test_parse_papers_empty_feed_returns_empty_list(self) -> None:
        """Test parsing a valid feed with no entries returns an empty list."""
        papers = parse_papers('<feed xmlns="http://www.w3.org/2005/Atom"></feed>')
        self.assertEqual(papers, [])

    def test_parse_papers_malformed_xml_raises(self) -> None:
        """Test malformed XML raises an explicit parse error."""
        with self.assertRaises(ArxivParseError):
            parse_papers("<feed>")


if __name__ == "__main__":
    unittest.main()
