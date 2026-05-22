import unittest
from unittest.mock import MagicMock

import numpy as np

from arxiv_recommender.arxiv_paper_fetcher.fetcher import ArxivFetcher
from arxiv_recommender.recommendation import RecommendationPipeline
from arxiv_recommender.schemas import AppConfig, Paper, VectorizerConfig
from arxiv_recommender.text_vectorization import TextEmbedder
from arxiv_recommender.utils import MetricsCollector


class TestRecommendationPipeline(unittest.TestCase):
    def setUp(self) -> None:
        self.config = AppConfig(
            favorite_papers_path="favorite_papers.json",
            vectorizer=VectorizerConfig(
                module_name="huggingface_embed",
                class_name="HuggingFaceEmbedding",
                model_name="distilbert-base-uncased",
                cache_size=1000,
            ),
            top_k=2,
        )
        self.favorite_papers = [
            Paper(title="Favorite 1", abstract="About machine learning."),
            Paper(title="Favorite 2", abstract="About neural networks."),
        ]
        self.daily_papers = [
            Paper(title="Candidate 1", abstract="About transformers."),
            Paper(title="Candidate 2", abstract="About optimization."),
            Paper(title="Candidate 3", abstract="About statistics."),
        ]
        self.mock_fetcher = MagicMock(spec=ArxivFetcher)
        self.mock_fetcher.get_daily_papers.return_value = self.daily_papers
        self.mock_vectorizer = MagicMock(spec=TextEmbedder)
        self.mock_vectorizer.process.side_effect = lambda text: np.array([len(text)])
        self.mock_vectorizer.get_cache_stats.return_value = {
            "hits": 1,
            "misses": 2,
            "hit_rate": 0.333,
            "size": 2,
        }
        self.metrics = MetricsCollector()
        self.favorite_loader = MagicMock(return_value=self.favorite_papers)
        self.favorite_prompt = MagicMock(return_value=self.favorite_papers)

    def _create_pipeline(self) -> RecommendationPipeline:
        return RecommendationPipeline(
            config=self.config,
            fetcher=self.mock_fetcher,
            vectorizer=self.mock_vectorizer,
            metrics=self.metrics,
            favorite_papers_loader=self.favorite_loader,
            favorite_papers_prompt=self.favorite_prompt,
        )

    def test_run_uses_existing_favorite_papers(self) -> None:
        pipeline = self._create_pipeline()

        result = pipeline.run()

        self.favorite_loader.assert_called_once_with(self.config.favorite_papers_path)
        self.favorite_prompt.assert_not_called()
        self.assertEqual(result.favorite_papers_count, 2)

    def test_run_prompts_when_favorite_papers_are_empty(self) -> None:
        self.favorite_loader.return_value = []
        pipeline = self._create_pipeline()

        result = pipeline.run()

        self.favorite_prompt.assert_called_once_with(
            self.config.favorite_papers_path, self.mock_fetcher
        )
        self.assertEqual(result.favorite_papers_count, 2)

    def test_run_respects_top_k(self) -> None:
        pipeline = self._create_pipeline()

        result = pipeline.run()

        self.assertEqual(len(result.recommendations), self.config.top_k)

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

    def test_run_reuses_injected_dependencies(self) -> None:
        pipeline = self._create_pipeline()

        result = pipeline.run(date_of_pulling_papers="20260522")

        self.mock_fetcher.get_daily_papers.assert_called_once_with(date="20260522")
        self.assertEqual(result.candidate_papers_count, len(self.daily_papers))
        self.assertGreater(self.mock_vectorizer.process.call_count, 0)


if __name__ == "__main__":
    unittest.main()
