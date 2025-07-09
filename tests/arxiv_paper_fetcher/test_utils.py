import unittest
from arxiv_recommender.arxiv_paper_fetcher.utils import format_arxiv_query, remove_control_characters


class TestUtils(unittest.TestCase):

    def test_format_arxiv_query(self):
        """Test formatting arXiv API query with category and max_results"""
        result = format_arxiv_query(category="cs.AI", max_results=100)
        self.assertIn("search_query=cat:cs.AI", result)
        self.assertIn("max_results=100", result)

    def test_format_arxiv_query_with_date(self):
        """Test formatting arXiv API query with date"""
        result = format_arxiv_query(date="20231001", category="cs.AI", max_results=100)
        self.assertIn("search_query=cat:cs.AI+AND+submittedDate:[202310010000+TO+202310012359]", result)
        self.assertIn("max_results=100", result)

    def test_remove_control_characters(self):
        """Test removing control characters from a string"""
        input_text = "Hello\nWorld\x00"
        cleaned_text = remove_control_characters(input_text)
        self.assertEqual(cleaned_text, "Hello World")

        input_text = "Hello\n World\x00"
        cleaned_text = remove_control_characters(input_text)
        self.assertEqual(cleaned_text, "Hello World")

        input_text = "\t Hello\n World\x00"
        cleaned_text = remove_control_characters(input_text)
        self.assertEqual(cleaned_text, "Hello World")


if __name__ == "__main__":
    unittest.main()
