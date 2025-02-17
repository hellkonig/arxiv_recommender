import unittest
import xml.etree.ElementTree as ET
from arxiv_recommender.arxiv_paper_fetcher.parser import parse_paper_info, parse_papers


class TestParser(unittest.TestCase):

    def setUp(self):
        """Set up test XML responses"""
        self.sample_entry = """
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <title>Sample Paper</title>
                <summary>Sample Abstract</summary>
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

    def test_parse_paper_info(self):
        """Test extracting title and abstract from a single entry"""
        paper = parse_paper_info(self.sample_entry)
        self.assertEqual(paper["title"], "Sample Paper")
        self.assertEqual(paper["abstract"], "Sample Abstract")

    def test_parse_papers(self):
        """Test extracting multiple papers from a feed"""
        papers = parse_papers(self.sample_feed)  # ✅ Now passing a string
        self.assertEqual(len(papers), 2)
        self.assertEqual(papers[0]["title"], "Paper 1")
        self.assertEqual(papers[1]["abstract"], "Abstract 2")


if __name__ == "__main__":
    unittest.main()
