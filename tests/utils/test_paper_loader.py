import unittest
import os

from arxiv_recommender.schemas import Paper
from arxiv_recommender.utils.json_handler import save_json
from arxiv_recommender.utils.paper_loader import load_favorite_papers


class TestPaperLoader(unittest.TestCase):
    """Test cases for loading favorite papers."""

    def setUp(self):
        """Setup sample data before each test."""
        self.sample_data = [{"title": "Sample Paper", "abstract": "Test abstract."}]
        save_json("test_favorite_papers.json", self.sample_data)
        self.expected_papers = [Paper(**p) for p in self.sample_data]

    def tearDown(self):
        """Cleanup test JSON file after tests."""
        if os.path.exists("test_favorite_papers.json"):
            os.remove("test_favorite_papers.json")
        if os.path.exists("non_existent.json"):
            os.remove("non_existent.json")

    def test_load_favorite_papers(self):
        """Test loading favorite papers from a JSON file."""
        result = load_favorite_papers("test_favorite_papers.json")
        self.assertEqual(result, self.expected_papers)

    def test_load_favorite_papers_file_not_found(self):
        """Test loading favorite papers when the file does not exist."""
        with self.assertLogs(level="INFO") as log:
            result = load_favorite_papers("non_existent.json")
            self.assertEqual(result, [])
            self.assertIn("Creating empty favorite papers file", log.output[1])