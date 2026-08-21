from time import perf_counter
from typing import ClassVar

import numpy as np
from pydantic import JsonValue
from sklearn.metrics.pairwise import cosine_similarity

from arxiv_recommender.provenance import ModelProvenance, SelectionSource
from arxiv_recommender.recommendation.types import RecommendationItem
from arxiv_recommender.schemas import Paper
from arxiv_recommender.text_vectorization import TextEmbedder
from arxiv_recommender.utils.metrics import MetricsCollector


class Recommender:
    """A content-based recommendation system for arXiv papers.

    Attributes:
        vectorizer: A text vectorization instance for computing embeddings.
        favorite_paper_embeddings: Precomputed embeddings for favorite papers.
        metrics: Optional metrics collector for observability.
    """

    # This metadata describes the scoring behavior implemented by this class;
    # it is not user configuration. Change it only with the corresponding
    # implementation, and bump the version whenever scoring behavior changes.
    RANKER_NAME: ClassVar[str] = "max_favorite_cosine_similarity"
    RANKER_VERSION: ClassVar[str] = "1.0.0"
    RANKER_CONFIG: ClassVar[dict[str, JsonValue]] = {
        "similarity": "cosine",
        "favorite_aggregation": "max",
        "sort_order": "descending",
        "tie_breaker": "candidate_input_order",
    }
    SELECTION_SOURCE: ClassVar[SelectionSource] = SelectionSource.BASE_RANKER

    def __init__(
        self,
        vectorizer: TextEmbedder,
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

    @property
    def provenance(self) -> ModelProvenance:
        """Return the developer-maintained base-ranker contract."""
        return ModelProvenance(
            name=self.RANKER_NAME,
            version=self.RANKER_VERSION,
            config=dict(self.RANKER_CONFIG),
        )

    @property
    def selection_source(self) -> SelectionSource:
        """Return the source assigned to recommendations from this ranker."""
        return self.SELECTION_SOURCE

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
            embedding = self.vectorizer.process(self.vectorizer.text_policy.build_text(paper))
            if self.metrics:
                self.metrics.add_embedding_latency(perf_counter() - start)
            embeddings.append(embedding)
        return np.array(embeddings)

    def recommend_by_papers(
        self, candidate_papers: list[Paper], top_k: int | None = None
    ) -> list[RecommendationItem]:
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
            embedding = self.vectorizer.process(self.vectorizer.text_policy.build_text(paper))
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
        ranked_papers = [
            RecommendationItem(
                paper=paper,
                score=float(score),
                selection_source=self.selection_source,
            )
            for paper, score in sorted_papers
        ]

        return ranked_papers[:top_k] if top_k else ranked_papers
