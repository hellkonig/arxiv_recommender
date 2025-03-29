import numpy as np
from typing import List, Dict, Optional
from sklearn.metrics.pairwise import cosine_similarity

from ..text_vectorization.distil_bert import DistilBERTEmbedding


class Recommender:
    """
    A content-based recommendation system for arXiv papers.

    Attributes:
        vectorizer (TextVectorization): A text vectorization instance for 
            computing embeddings.
        favorite_paper_embeddings (np.ndarray): Precomputed embeddings for
            favorite papers.
    """

    def __init__(
        self, 
        vectorizer: DistilBERTEmbedding,
        favorite_papers: List[Dict[str, str]]
    ) -> None:
        """
        Initializes the recommender with a text vectorization model and 
        favorite paper metadata.

        Args:
            vectorizer (TextVectorization): An instance of the text 
                vectorization class.
            favorite_papers (List[Dict[str, str]]): A list of favorite papers, 
                each containing "title" and "abstract".

        Raises:
            ValueError: If no favorite papers are provided.
        """
        if not favorite_papers:
            raise ValueError("At least one favorite paper must be provided.")

        self.vectorizer = vectorizer
        self.favorite_paper_embeddings = self._compute_favorite_embeddings(
            favorite_papers
        )

    def _compute_favorite_embeddings(
            self,
            papers: List[Dict[str, str]]
        ) -> np.ndarray:
        """
        Computes embeddings for the user's favorite papers.

        Args:
            papers (List[Dict[str, str]]): A list of favorite papers, each
                containing "title" and "abstract".

        Returns:
            np.ndarray: An array of embeddings for the favorite papers.
        """
        return np.array([
            self.vectorizer.process(paper["title"] + " " + paper["abstract"])
            for paper in papers
        ])

    def recommend_by_papers(
        self, 
        candidate_papers: List[Dict[str, str]], 
        **kwargs
    ) -> List[Dict[str, str]]:
        """
        Recommends papers based on the highest similarity to the user's 
        favorite papers.

        Args:
            candidate_papers (List[Dict[str, str]]): A list of candidate 
                papers, each containing "title" and "abstract".
            **kwargs:
                top_k (Optional[int]): The number of top-ranked papers to 
                    return. If not provided, returns all ranked papers.

        Returns:
            List[Dict[str, str]]: A ranked list of recommended papers, 
                sorted by highest similarity.
        """
        top_k = kwargs.get("top_k")

        if self.favorite_paper_embeddings.size == 0 or not candidate_papers:
            return []

        # Compute embeddings for candidate papers
        candidate_embeddings = np.array([
            self.vectorizer.process(
                paper["title"] + " " + paper["abstract"]
            ) 
            for paper in candidate_papers
        ])

        # Compute cosine similarity between favorite and candidate papers
        similarity_matrix = cosine_similarity(
            candidate_embeddings, self.favorite_paper_embeddings
        )

        # Use the maximum similarity score for each candidate paper
        max_similarities = similarity_matrix.max(axis=1)

        # Rank candidate papers by similarity (descending order)
        ranked_papers = sorted(
            zip(candidate_papers, max_similarities),
            key=lambda x: x[1],
            reverse=True
        )

        # Extract ranked papers with similarity scores
        ranked_papers = [
            {
                "title": paper["title"], 
                "abstract": paper["abstract"], 
                "score": score
            }
            for paper, score in ranked_papers
        ]

        return ranked_papers[:top_k] if top_k else ranked_papers
