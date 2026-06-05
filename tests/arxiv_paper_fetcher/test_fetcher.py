import unittest
from unittest.mock import Mock, patch

import requests
from arxiv_recommender.arxiv_paper_fetcher.fetcher import ArxivFetcher
from arxiv_recommender.arxiv_paper_fetcher.parser import ArxivParseError


class TestArxivFetcher(unittest.TestCase):
    def setUp(self) -> None:
        """Initialize the ArxivFetcher instance for tests."""
        self.fetcher = ArxivFetcher(max_results=5)

    def _mock_response(self, text: str, status_code: int = 200) -> Mock:
        """Create a response mock with realistic status handling."""
        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.text = text
        mock_response.raise_for_status.return_value = None
        return mock_response

    @patch("requests.get")
    def test_get_paper_by_id_success(self, mock_get: Mock) -> None:
        """Test fetching a single paper successfully by ID."""
        mock_response = self._mock_response("""<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <title>Sample Paper</title>
                <summary>Sample Abstract</summary>
            </entry>
        </feed>""")
        mock_get.return_value = mock_response

        paper = self.fetcher.get_paper_by_id("1234.56789")
        assert paper is not None
        self.assertEqual(paper.title, "Sample Paper")
        self.assertEqual(paper.abstract, "Sample Abstract")
        mock_get.assert_called_once_with(
            "http://export.arxiv.org/api/query",
            params={"id_list": "1234.56789"},
            timeout=self.fetcher.timeout,
        )
        mock_response.raise_for_status.assert_called_once()

    @patch("arxiv_recommender.utils.retry.time.sleep", return_value=None)
    @patch("requests.get")
    def test_get_paper_by_id_http_error(self, mock_get: Mock, mock_sleep: Mock) -> None:
        """Test HTTP errors raise after the retry policy is applied."""
        mock_response = self._mock_response("", status_code=404)
        mock_response.raise_for_status.side_effect = requests.HTTPError("404 Client Error")
        mock_get.return_value = mock_response

        with self.assertRaises(requests.HTTPError):
            self.fetcher.get_paper_by_id("non_existent_paper")

        self.assertEqual(mock_get.call_count, 4)
        self.assertEqual(mock_sleep.call_count, 3)

    @patch("requests.get")
    def test_get_paper_by_id_empty_feed_returns_none(self, mock_get: Mock) -> None:
        """Test a successful empty feed returns None."""
        mock_get.return_value = self._mock_response(
            """<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <!-- No entries -->
        </feed>"""
        )

        paper = self.fetcher.get_paper_by_id("missing_paper")
        self.assertIsNone(paper)

    @patch("requests.get")
    def test_get_paper_by_id_malformed_xml_raises(self, mock_get: Mock) -> None:
        """Test malformed XML raises an explicit parse error."""
        mock_get.return_value = self._mock_response("<feed>")

        with self.assertRaises(ArxivParseError):
            self.fetcher.get_paper_by_id("1234.56789")

    @patch("arxiv_recommender.utils.retry.time.sleep", return_value=None)
    @patch("requests.get")
    def test_get_paper_by_id_network_error(self, mock_get: Mock, mock_sleep: Mock) -> None:
        """Test network errors raise after the retry policy is applied."""
        mock_get.side_effect = requests.RequestException("Network error")

        with self.assertRaises(requests.RequestException):
            self.fetcher.get_paper_by_id("1234.56789")

        self.assertEqual(mock_get.call_count, 4)
        self.assertEqual(mock_sleep.call_count, 3)

    @patch("requests.get")
    def test_get_daily_papers_success(self, mock_get: Mock) -> None:
        """Test fetching new daily papers successfully."""
        mock_response = self._mock_response("""<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <entry>
                <title>Paper 1</title>
                <summary>Abstract 1</summary>
            </entry>
            <entry>
                <title>Paper 2</title>
                <summary>Abstract 2</summary>
            </entry>
        </feed>""")
        mock_get.return_value = mock_response

        papers = self.fetcher.get_daily_papers(date="20231001", category="cs.AI")
        self.assertEqual(len(papers), 2)
        self.assertEqual(papers[0].title, "Paper 1")
        self.assertEqual(papers[1].abstract, "Abstract 2")
        mock_get.assert_called_once_with(
            "http://export.arxiv.org/api/query",
            params={
                "search_query": "cat:cs.AI AND submittedDate:[202310010000 TO 202310012359]",
                "max_results": self.fetcher.max_results,
            },
            timeout=self.fetcher.timeout,
        )
        mock_response.raise_for_status.assert_called_once()

    @patch("requests.get")
    def test_get_daily_papers_no_results(self, mock_get: Mock) -> None:
        """Test fetching daily papers when no new papers are available."""
        mock_response = self._mock_response("""<?xml version="1.0" encoding="UTF-8"?>
        <feed xmlns="http://www.w3.org/2005/Atom">
            <!-- No entries -->
        </feed>""")
        mock_get.return_value = mock_response

        papers = self.fetcher.get_daily_papers(date="20231001", category="cs.LG")
        self.assertEqual(len(papers), 0)
        mock_get.assert_called_once_with(
            "http://export.arxiv.org/api/query",
            params={
                "search_query": "cat:cs.LG AND submittedDate:[202310010000 TO 202310012359]",
                "max_results": self.fetcher.max_results,
            },
            timeout=self.fetcher.timeout,
        )

    @patch("requests.get")
    def test_get_daily_papers_malformed_xml_raises(self, mock_get: Mock) -> None:
        """Test malformed XML raises an explicit parse error."""
        mock_get.return_value = self._mock_response("<feed>")

        with self.assertRaises(ArxivParseError):
            self.fetcher.get_daily_papers(category="cs.LG")

    @patch("arxiv_recommender.utils.retry.time.sleep", return_value=None)
    @patch("requests.get")
    def test_get_daily_papers_http_error(self, mock_get: Mock, mock_sleep: Mock) -> None:
        """Test HTTP errors raise after the retry policy is applied."""
        mock_response = self._mock_response("", status_code=500)
        mock_response.raise_for_status.side_effect = requests.HTTPError("500 Server Error")
        mock_get.return_value = mock_response

        with self.assertRaises(requests.HTTPError):
            self.fetcher.get_daily_papers(category="cs.LG")

        self.assertEqual(mock_get.call_count, 4)
        self.assertEqual(mock_sleep.call_count, 3)

    @patch("arxiv_recommender.utils.retry.time.sleep", return_value=None)
    @patch("requests.get")
    def test_get_daily_papers_network_error(self, mock_get: Mock, mock_sleep: Mock) -> None:
        """Test network errors raise after the retry policy is applied."""
        mock_get.side_effect = requests.RequestException("API timeout")

        with self.assertRaises(requests.RequestException):
            self.fetcher.get_daily_papers(category="cs.LG")

        self.assertEqual(mock_get.call_count, 4)
        self.assertEqual(mock_sleep.call_count, 3)


if __name__ == "__main__":
    unittest.main()
