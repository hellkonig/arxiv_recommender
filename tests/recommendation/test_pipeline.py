import unittest
from unittest.mock import MagicMock

import numpy as np

from arxiv_recommender.arxiv_paper_fetcher.fetcher import ArxivFetcher
from arxiv_recommender.favorite_papers import FavoritePapersProvider
from arxiv_recommender.provenance import ModelProvenance
from arxiv_recommender.recommendation import RecommendationPipeline
from arxiv_recommender.schemas import AppConfig, Paper, VectorizerConfig
from arxiv_recommender.text_vectorization import TextEmbedder
from arxiv_recommender.text_vectorization.text_policy import TitleAbstractTextPolicy
from arxiv_recommender.utils import MetricsCollector


class TestRecommendationPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig(
            favorite_papers_path="favorite_papers.json",
            vectorizer=VectorizerConfig(
                module_name="huggingface_embed",
                class_name="HuggingFaceEmbedding",
                model_name="distilbert-base-uncased",
                model_revision="a" * 40,
                cache_size=1000,
            ),
            top_k=2,
        )
        self.favorite_papers = [
            Paper(title="Favorite 1", abstract="About machine learning."),
            Paper(title="Favorite 2", abstract="About neural networks."),
        ]
        self.daily_papers = [
            Paper(
                arxiv_id="1234.56789",
                url="http://arxiv.org/abs/1234.56789",
                title="Candidate 1",
                abstract="About transformers.",
            ),
            Paper(title="Candidate 2", abstract="About optimization."),
            Paper(title="Candidate 3", abstract="About statistics."),
        ]
        self.mock_fetcher = MagicMock(spec=ArxivFetcher)
        self.mock_fetcher.get_daily_papers.return_value = self.daily_papers
        self.mock_vectorizer = MagicMock(spec=TextEmbedder)
        self.mock_vectorizer.process.side_effect = lambda text: np.array([len(text)])
        self.mock_vectorizer.text_policy = TitleAbstractTextPolicy()
        self.mock_vectorizer.get_cache_stats.return_value = {
            "hits": 1,
            "misses": 2,
            "hit_rate": 0.333,
            "size": 2,
        }
        self.embedding_provenance = ModelProvenance(
            name="distilbert-base-uncased",
            version="1.0.0",
            config={
                "pooling_strategy": "mean",
                "normalize_embeddings": False,
                "max_length": 512,
                "text_policy": {
                    "name": "title_abstract",
                    "version": "1.0.0",
                    "config": {"separator": " "},
                },
            },
        )
        self.ranker_provenance = ModelProvenance(
            name="max_favorite_cosine_similarity",
            version="1.0.0",
            config={
                "similarity": "cosine",
                "favorite_aggregation": "max",
                "sort_order": "descending",
                "tie_breaker": "candidate_input_order",
            },
        )
        self.mock_vectorizer.provenance = self.embedding_provenance
        self.metrics = MetricsCollector()
        self.favorite_papers_provider = MagicMock(spec=FavoritePapersProvider)
        self.favorite_papers_provider.get_papers.return_value = self.favorite_papers

    def _create_pipeline(self) -> RecommendationPipeline:
        return RecommendationPipeline(
            config=self.config,
            fetcher=self.mock_fetcher,
            vectorizer=self.mock_vectorizer,
            metrics=self.metrics,
            favorite_papers_provider=self.favorite_papers_provider,
        )

    def test_run_uses_provider_favorite_papers(self) -> None:
        pipeline = self._create_pipeline()

        result = pipeline.run()

        self.favorite_papers_provider.get_papers.assert_called_once_with()
        self.assertEqual(result.favorite_papers_count, 2)

    def test_run_raises_if_provider_returns_empty_papers(self) -> None:
        self.favorite_papers_provider.get_papers.return_value = []
        pipeline = self._create_pipeline()

        with self.assertRaises(ValueError):
            pipeline.run()

    def test_run_respects_top_k(self) -> None:
        pipeline = self._create_pipeline()

        result = pipeline.run()

        self.assertEqual(len(result.recommendations), self.config.top_k)

    def test_run_preserves_recommendation_metadata(self) -> None:
        """RecommendationPipeline should not drop metadata while orchestrating ranking."""
        pipeline = self._create_pipeline()

        result = pipeline.run()

        metadata_recommendation = next(
            item for item in result.recommendations if item.paper.arxiv_id == "1234.56789"
        )
        self.assertEqual(metadata_recommendation.paper, self.daily_papers[0])

    def test_run_returns_empty_recommendations_for_empty_daily_papers(self) -> None:
        self.mock_fetcher.get_daily_papers.return_value = []
        pipeline = self._create_pipeline()

        result = pipeline.run()

        self.assertEqual(result.recommendations, [])
        self.assertEqual(result.candidate_papers_count, 0)

    def test_run_includes_cache_stats_in_metrics_summary(self) -> None:
        pipeline = self._create_pipeline()

        result = pipeline.run()

        self.assertIn("cache", result.metrics_summary)
        self.assertEqual(result.metrics_summary["cache"], self.mock_vectorizer.get_cache_stats())

    def test_run_exposes_resolved_model_provenance(self) -> None:
        pipeline = self._create_pipeline()

        result = pipeline.run()

        self.assertEqual(result.embedding_provenance, self.embedding_provenance)
        self.assertEqual(result.ranker_provenance, self.ranker_provenance)

    def test_run_reuses_injected_dependencies(self) -> None:
        pipeline = self._create_pipeline()

        result = pipeline.run(date_of_pulling_papers="20260522")

        self.mock_fetcher.get_daily_papers.assert_called_once_with(date="20260522")
        self.assertEqual(result.candidate_papers_count, len(self.daily_papers))
        self.assertGreater(self.mock_vectorizer.process.call_count, 0)


if __name__ == "__main__":
    unittest.main()
