from abc import ABC, abstractmethod

from arxiv_recommender.schemas import Paper


class FavoritePapersProvider(ABC):
    """Provides favorite papers for a recommendation run."""

    @abstractmethod
    def get_papers(self) -> list[Paper]:
        """Return favorite papers for the current run."""
