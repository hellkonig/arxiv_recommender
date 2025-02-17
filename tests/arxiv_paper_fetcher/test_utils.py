import unittest
from arxiv_recommender.arxiv_paper_fetcher.utils import format_arxiv_query


class TestUtils(unittest.TestCase):

    def test_format_arxiv_query(self):
        """Test formatting arXiv API query with category and max_results"""
        result = format_arxiv_query("cs.AI", max_results=100)
        self.assertIn("search_query=cat:cs.AI", result)
        self.assertIn("max_results=100", result)


if __name__ == "__main__":
    unittest.main()
