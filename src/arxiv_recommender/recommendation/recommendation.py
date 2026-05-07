from time import perf_counter

import numpy as np
from typing import Any
from sklearn.metrics.pairwise import cosine_similarity

from arxiv_recommender.schemas import Paper
from arxiv_recommender.utils.metrics import MetricsCollector
from ..text_vectorization import DistilBERTEmbedding


class Recommender:
    """A content-based recommendation system for arXiv papers.

    Attributes:
        vectorizer: A text vectorization instance for computing embeddings.
        favorite_paper_embeddings: Precomputed embeddings for favorite papers.
        metrics: Optional metrics collector for observability.
    """

    def __init__(
        self,
        vectorizer: DistilBERTEmbedding,
        favorite_papers: list[Paper],
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Initializes the recommender with a text vectorization model.

        Args:
            vectorizer: An instance of the text vectorization class.
            favorite_papers: A list of favorite papers, each containing
                "title" and "abstract".
            metrics: Optional metrics collector for tracking performance.

        Raises:
            ValueError: If no favorite papers are provided.
        """
        if not favorite_papers:
            raise ValueError("At least one favorite paper must be provided.")

        self.vectorizer = vectorizer
        self.metrics = metrics
        self.favorite_paper_embeddings = self._compute_favorite_embeddings(favorite_papers)

    def _compute_favorite_embeddings(self, papers: list[Paper]) -> np.ndarray:
        """Computes embeddings for the user's favorite papers.

        Args:
            papers: A list of favorite papers, each containing
                "title" and "abstract".

        Returns:
            An array of embeddings for the favorite papers.
        """
        embeddings = []
        for paper in papers:
            start = perf_counter()
            embedding = self.vectorizer.process(paper.title + " " + paper.abstract)
            if self.metrics:
                self.metrics.add_embedding_latency(perf_counter() - start)
            embeddings.append(embedding)
        return np.array(embeddings)

    def recommend_by_papers(
        self, candidate_papers: list[Paper], top_k: int | None = None
    ) -> list[dict[str, Any]]:
        """Recommends papers based on the highest similarity to favorite papers.

        Args:
            candidate_papers: A list of candidate papers, each containing
                "title" and "abstract".
            top_k: The number of top-ranked papers to return. If not provided,
                returns all ranked papers.

        Returns:
            A ranked list of recommended papers, sorted by highest similarity.
        """
        if self.favorite_paper_embeddings.size == 0 or not candidate_papers:
            return []

        embedding_list = []
        for paper in candidate_papers:
            start = perf_counter()
            embedding = self.vectorizer.process(paper.title + " " + paper.abstract)
            if self.metrics:
                self.metrics.add_embedding_latency(perf_counter() - start)
            embedding_list.append(embedding)
        candidate_embeddings = np.array(embedding_list)

        # Compute cosine similarity between favorite and candidate papers
        similarity_matrix = cosine_similarity(candidate_embeddings, self.favorite_paper_embeddings)

        # Use the maximum similarity score for each candidate paper
        max_similarities = similarity_matrix.max(axis=1)

        # Rank candidate papers by similarity (descending order)
        sorted_papers: list[tuple[Paper, float]] = sorted(
            zip(candidate_papers, max_similarities), key=lambda x: x[1], reverse=True
        )

        # Extract ranked papers with similarity scores
        ranked_papers: list[dict[str, Any]] = [
            {"title": paper.title, "abstract": paper.abstract, "score": float(score)}
            for paper, score in sorted_papers
        ]

        return ranked_papers[:top_k] if top_k else ranked_papers
