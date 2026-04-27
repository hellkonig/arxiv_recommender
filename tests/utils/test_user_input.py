import unittest
from unittest.mock import patch, MagicMock

from arxiv_recommender.schemas import Paper
from arxiv_recommender.utils.user_input import get_favorite_papers_from_user


class TestUserInput(unittest.TestCase):
    """Test cases for user-provided arXiv paper input."""

    @patch("arxiv_recommender.utils.user_input.save_json")
    @patch("builtins.input", side_effect=["1234.5678", "9876.5432", ""])
    def test_get_favorite_papers_from_user(self, mock_input, mock_save_json):
        """
        Test user input collection and metadata retrieval.
        """
        mock_fetcher = MagicMock()
        mock_fetcher.get_paper_by_id.side_effect = [
            Paper(title="Paper 1", abstract="Abstract 1"),
            Paper(title="Paper 2", abstract="Abstract 2"),
        ]

        output_file = "favorite_papers.json"
        papers = get_favorite_papers_from_user(output_file, mock_fetcher)

        self.assertEqual(len(papers), 2)
        self.assertEqual(papers[0].title, "Paper 1")
        self.assertEqual(papers[1].title, "Paper 2")

        mock_save_json.assert_called_once_with(output_file, [p.model_dump() for p in papers])

    @patch("arxiv_recommender.utils.json_handler.save_json")
    @patch("builtins.input", side_effect=[""])
    def test_empty_input_raises_error(self, mock_input, mock_save_json):
        """
        Test that an error is raised when no valid papers are entered.
        """
        mock_fetcher = MagicMock()
        mock_fetcher.get_paper_by_id.side_effect = []

        with self.assertRaises(ValueError) as context:
            get_favorite_papers_from_user("dummy.json", mock_fetcher)

        self.assertEqual(str(context.exception), "At least one valid arXiv paper is required.")

        mock_save_json.assert_not_called()

    @patch("arxiv_recommender.utils.json_handler.save_json")
    @patch("builtins.input", side_effect=["invalid_id", ""])
    def test_invalid_paper_id_handling(self, mock_input, mock_save_json):
        """
        Test behavior when an invalid paper ID is entered.
        """
        mock_fetcher = MagicMock()
        mock_fetcher.get_paper_by_id.return_value = None

        with self.assertRaises(ValueError):
            get_favorite_papers_from_user("dummy.json", mock_fetcher)

        mock_save_json.assert_not_called()


if __name__ == "__main__":
    unittest.main()
