from arxiv_recommender.arxiv_paper_fetcher.fetcher import ArxivFetcher
from arxiv_recommender.favorite_papers.base import FavoritePapersProvider
from arxiv_recommender.favorite_papers.loader import load_favorite_papers
from arxiv_recommender.favorite_papers.user_input import get_favorite_papers_from_user
from arxiv_recommender.schemas import Paper


class FileFavoritePapersProvider(FavoritePapersProvider):
    """Loads favorite papers from disk with CLI fallback prompting."""

    def __init__(self, favorite_papers_path: str, fetcher: ArxivFetcher) -> None:
        self._favorite_papers_path = favorite_papers_path
        self._fetcher = fetcher

    def get_papers(self) -> list[Paper]:
        """Return favorite papers from disk, prompting if none are stored."""
        favorite_papers = load_favorite_papers(self._favorite_papers_path)
        if favorite_papers:
            return favorite_papers
        return get_favorite_papers_from_user(self._favorite_papers_path, self._fetcher)
