import unittest
import numpy as np
from unittest.mock import MagicMock

from arxiv_recommender.recommendation.recommendation import Recommender
from arxiv_recommender.text_vectorization.distil_bert import DistilBERTEmbedding


class TestRecommender(unittest.TestCase):
    """Unit tests for the Recommender class."""

    def setUp(self):
        """Setup mock vectorizer and sample papers for testing."""
        # Mock vectorizer with a simple embedding function
        self.mock_vectorizer = MagicMock(spec=DistilBERTEmbedding)
        self.mock_vectorizer.process.side_effect = lambda text: np.array(
            [len(text)]  # Simple mock embedding: vector length is text length
        )

        # Sample favorite papers
        self.favorite_papers = [
            {"title": "AI Research", "abstract": "This paper discusses AI."},
            {"title": "Quantum Computing", "abstract": "Quantum mechanics applied to computing."}
        ]

        # Sample candidate papers
        self.candidate_papers = [
            {"title": "Deep Learning", "abstract": "Neural networks and optimization."},
            {"title": "Machine Learning", "abstract": "ML techniques and applications."},
            {"title": "Quantum AI", "abstract": "AI techniques for quantum systems."}
        ]

        # Initialize recommender
        self.recommender = Recommender(
            vectorizer=self.mock_vectorizer,
            favorite_papers=self.favorite_papers
        )

    def test_favorite_paper_embeddings_computed_correctly(self):
        """Test if favorite paper embeddings are computed correctly."""
        expected_embeddings = np.array([[len(p["title"] + " " + p["abstract"])]
                                        for p in self.favorite_papers])
        np.testing.assert_array_equal(
            self.recommender.favorite_paper_embeddings, expected_embeddings
        )

    def test_recommend_by_papers_with_valid_candidates(self):
        """Test if recommendations are correctly ranked by similarity."""
        recommendations = self.recommender.recommend_by_papers(self.candidate_papers)

        self.assertTrue(len(recommendations) > 0)
        self.assertEqual(len(recommendations), len(self.candidate_papers))
        self.assertEqual(
            sorted(recommendations, key=lambda x: x["score"], reverse=True),
            recommendations
        )  # Ensure sorting by similarity

    def test_recommend_by_papers_with_top_k(self):
        """Test if top_k parameter limits the results correctly."""
        recommendations = self.recommender.recommend_by_papers(
            self.candidate_papers, top_k=2
        )
        self.assertEqual(len(recommendations), 2)

    def test_recommend_by_papers_with_empty_candidates(self):
        """Test behavior when no candidate papers are provided."""
        recommendations = self.recommender.recommend_by_papers([])
        self.assertEqual(recommendations, [])

    def test_init_raises_error_if_no_favorites(self):
        """Test that an error is raised when no favorite papers are provided."""
        with self.assertRaises(ValueError):
            Recommender(vectorizer=self.mock_vectorizer, favorite_papers=None)

        with self.assertRaises(ValueError):
            Recommender(vectorizer=self.mock_vectorizer, favorite_papers=[])

if __name__ == "__main__":
    unittest.main()
