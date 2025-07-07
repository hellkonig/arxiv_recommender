import unittest
from arxiv_recommender.arxiv_paper_fetcher.utils import format_arxiv_query, remove_control_characters


class TestUtils(unittest.TestCase):

    def test_format_arxiv_query(self):
        """Test formatting arXiv API query with category and max_results"""
        result = format_arxiv_query("cs.AI", max_results=100)
        self.assertIn("search_query=cat:cs.AI", result)
        self.assertIn("max_results=100", result)

    def test_remove_control_characters(self):
        """Test removing control characters from a string"""
        input_text = "Hello\nWorld\x00"
        cleaned_text = remove_control_characters(input_text)
        self.assertEqual(cleaned_text, "HelloWorld")

        input_text = "Hello\n World\x00"
        cleaned_text = remove_control_characters(input_text)
        self.assertEqual(cleaned_text, "Hello World")

if __name__ == "__main__":
    unittest.main()
