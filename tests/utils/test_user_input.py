import unittest
from unittest.mock import patch, mock_open
import json
import os
from arxiv_recommender.utils.user_input import get_favorite_papers_from_user

TEST_JSON_FILE = "test_favorite_papers.json"


class TestUserInput(unittest.TestCase):
    """Test cases for user input handling."""

    def setUp(self):
        """Ensure test file is removed before each test."""
        if os.path.exists(TEST_JSON_FILE):
            os.remove(TEST_JSON_FILE)

    def tearDown(self):
        """Cleanup test file after each test."""
        if os.path.exists(TEST_JSON_FILE):
            os.remove(TEST_JSON_FILE)

    @patch("builtins.input", side_effect=["1234.5678", ""])
    @patch("utils.user_input.fetch_paper_metadata")
    @patch("builtins.open", new_callable=mock_open)
    def test_get_favorite_papers_from_user(
        self, mock_file, mock_fetch, mock_input
    ):
        """
        Test user input for arXiv paper IDs and verify JSON storage.
        """

        # Mock API response
        mock_fetch.return_value = [
            {"id": "1234.5678", "title": "Test Paper", "abstract": "Test abstract"}
        ]

        get_favorite_papers_from_user(TEST_JSON_FILE)

        # Ensure file was written
        mock_file.assert_called_once_with(TEST_JSON_FILE, "w", encoding="utf-8")

        # Verify file contents
        with open(TEST_JSON_FILE, "r", encoding="utf-8") as file:
            data = json.load(file)
            self.assertEqual(len(data), 1)
            self.assertEqual(data[0]["id"], "1234.5678")
            self.assertEqual(data[0]["title"], "Test Paper")

    @patch("builtins.input", side_effect=[""])
    def test_no_input_provided(self, mock_input):
        """
        Test case when no IDs are provided.
        Expect an error to be raised.
        """
        with self.assertRaises(ValueError):
            get_favorite_papers_from_user(TEST_JSON_FILE)


if __name__ == "__main__":
    unittest.main()
