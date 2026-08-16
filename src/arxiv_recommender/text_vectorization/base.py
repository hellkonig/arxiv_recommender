from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from arxiv_recommender.provenance import ModelProvenance


class TextEmbedder(ABC):
    """Abstract base class for text embedding models.

    All text vectorization implementations should inherit from this class
    and implement the ``provenance``, ``process``, and ``get_cache_stats``
    contracts.
    """

    @property
    @abstractmethod
    def provenance(self) -> ModelProvenance:
        """Return the resolved model identity and embedding configuration."""

    @abstractmethod
    def process(self, text: str) -> np.ndarray:
        """Generate an embedding vector for the given text.

        Args:
            text: Input text string.

        Returns:
            A numpy array representing the text embedding.
        """

    @abstractmethod
    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Dictionary with cache hits, misses, and hit rate.
        """
