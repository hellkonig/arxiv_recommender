import unittest
from unittest.mock import MagicMock, patch

from arxiv_recommender.arxiv_paper_fetcher.fetcher import ArxivFetcher
from arxiv_recommender.favorite_papers import FileFavoritePapersProvider
from arxiv_recommender.schemas import Paper


class TestFileFavoritePapersProvider(unittest.TestCase):
    def setUp(self) -> None:
        self.favorite_papers_path = "favorite_papers.json"
        self.fetcher = MagicMock(spec=ArxivFetcher)
        self.favorite_papers = [
            Paper(title="Favorite 1", abstract="About machine learning."),
            Paper(title="Favorite 2", abstract="About neural networks."),
        ]

    @patch("arxiv_recommender.favorite_papers.file_provider.get_favorite_papers_from_user")
    @patch("arxiv_recommender.favorite_papers.file_provider.load_favorite_papers")
    def test_get_papers_returns_loaded_papers(
        self,
        mock_load_favorite_papers: MagicMock,
        mock_get_favorite_papers_from_user: MagicMock,
    ) -> None:
        mock_load_favorite_papers.return_value = self.favorite_papers
        provider = FileFavoritePapersProvider(self.favorite_papers_path, self.fetcher)

        result = provider.get_papers()

        self.assertEqual(result, self.favorite_papers)
        mock_load_favorite_papers.assert_called_once_with(self.favorite_papers_path)
        mock_get_favorite_papers_from_user.assert_not_called()

    @patch("arxiv_recommender.favorite_papers.file_provider.get_favorite_papers_from_user")
    @patch("arxiv_recommender.favorite_papers.file_provider.load_favorite_papers")
    def test_get_papers_prompts_when_loaded_papers_are_empty(
        self,
        mock_load_favorite_papers: MagicMock,
        mock_get_favorite_papers_from_user: MagicMock,
    ) -> None:
        mock_load_favorite_papers.return_value = []
        mock_get_favorite_papers_from_user.return_value = self.favorite_papers
        provider = FileFavoritePapersProvider(self.favorite_papers_path, self.fetcher)

        result = provider.get_papers()

        self.assertEqual(result, self.favorite_papers)
        mock_load_favorite_papers.assert_called_once_with(self.favorite_papers_path)
        mock_get_favorite_papers_from_user.assert_called_once_with(
            self.favorite_papers_path, self.fetcher
        )


if __name__ == "__main__":
    unittest.main()
