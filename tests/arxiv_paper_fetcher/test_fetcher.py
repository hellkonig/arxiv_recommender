import unittest
from unittest.mock import patch, Mock
import requests
from arxiv_recommender.arxiv_paper_fetcher.fetcher import ArxivFetcher


class TestArxivFetcher(unittest.TestCase):

    def setUp(self):
        """Initialize the ArxivFetcher instance for tests."""
        self.fetcher = ArxivFetcher(max_results=5)

    @patch("requests.get")
    def test_get_paper_by_id_success(self, mock_get):
        """Test fetching a single paper successfully by ID."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <title>Sample Paper</title>
                <summary>Sample Abstract</summary>
            </entry>
        </feed>"""
        mock_get.return_value = mock_response

        paper = self.fetcher.get_paper_by_id("1234.56789")
        self.assertIsNotNone(paper)
        self.assertEqual(paper["title"], "Sample Paper")
        self.assertEqual(paper["abstract"], "Sample Abstract")

    @patch("requests.get")
    def test_get_paper_by_id_failure(self, mock_get):
        """Test handling of a failed paper fetch (e.g., paper not found)."""
        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.text = ""  # Empty response
        mock_get.return_value = mock_response

        paper = self.fetcher.get_paper_by_id("non_existent_paper")
        self.assertIsNone(paper)

    @patch("requests.get")
    def test_get_paper_by_id_network_error(self, mock_get):
        """Test handling of network errors when fetching a paper."""
        mock_get.side_effect = requests.RequestException("Network error")
        
        with self.assertRaises(requests.RequestException):
            self.fetcher.get_paper_by_id("1234.56789")

    @patch("requests.get")
    def test_get_daily_papers_success(self, mock_get):
        """Test fetching new daily papers successfully."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <title>Paper 1</title>
                <summary>Abstract 1</summary>
            </entry>
            <entry>
                <title>Paper 2</title>
                <summary>Abstract 2</summary>
            </entry>
        </feed>"""
        mock_get.return_value = mock_response

        papers = self.fetcher.get_daily_papers(category="cs.AI")
        self.assertEqual(len(papers), 2)
        self.assertEqual(papers[0]["title"], "Paper 1")
        self.assertEqual(papers[1]["abstract"], "Abstract 2")

    @patch("requests.get")
    def test_get_daily_papers_no_results(self, mock_get):
        """Test fetching daily papers when no new papers are available."""
        mock_response = Mock()
        mock_response.status_code = 200
        mock_response.text = """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <!-- No entries -->
        </feed>"""
        mock_get.return_value = mock_response

        papers = self.fetcher.get_daily_papers(category="cs.LG")
        self.assertEqual(len(papers), 0)

    @patch("requests.get")
    def test_get_daily_papers_network_error(self, mock_get):
        """Test handling of network errors when fetching daily papers."""
        mock_get.side_effect = requests.RequestException("API timeout")

        with self.assertRaises(requests.RequestException):
            self.fetcher.get_daily_papers(category="cs.LG")


if __name__ == "__main__":
    unittest.main()
