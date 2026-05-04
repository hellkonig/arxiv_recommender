import hashlib
from collections import OrderedDict
from typing import Any

import numpy as np


class EmbeddingCache:
    """
    LRU cache for storing text embeddings.

    Attributes:
        max_size (int): Maximum number of embeddings to store.
        _cache (OrderedDict): Internal cache storage.
    """

    def __init__(self, max_size: int = 1000) -> None:
        """
        Initializes the embedding cache.

        Args:
            max_size (int): Maximum number of embeddings to store.
        """
        self.max_size = max_size
        self._cache: OrderedDict[str, np.ndarray] = OrderedDict()
        self._hits = 0
        self._misses = 0

    def _compute_key(self, text: str) -> str:
        """
        Computes a hash key for the given text.

        Args:
            text (str): Input text.

        Returns:
            str: SHA256 hash of the text.
        """
        return hashlib.sha256(text.encode()).hexdigest()

    def get(self, text: str) -> np.ndarray | None:
        """
        Retrieves an embedding from the cache.

        Args:
            text (str): Input text.

        Returns:
            np.ndarray | None: Cached embedding or None if not found.
        """
        key = self._compute_key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
            self._hits += 1
            return self._cache[key]
        self._misses += 1
        return None

    def put(self, text: str, embedding: np.ndarray) -> None:
        """
        Stores an embedding in the cache.

        Args:
            text (str): Input text.
            embedding (np.ndarray): Embedding to store.
        """
        key = self._compute_key(text)
        if key in self._cache:
            self._cache.move_to_end(key)
        self._cache[key] = embedding.copy()
        if len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    def clear(self) -> None:
        """Clears all cached embeddings and resets stats."""
        self._cache.clear()
        self._hits = 0
        self._misses = 0

    def stats(self) -> dict[str, Any]:
        """
        Returns cache statistics.

        Returns:
            dict[str, Any]: Dictionary with hits, misses, size, and max_size.
        """
        total = self._hits + self._misses
        hit_rate = self._hits / total if total > 0 else 0.0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "size": len(self._cache),
            "max_size": self.max_size,
            "hit_rate": hit_rate,
        }