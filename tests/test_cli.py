import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from arxiv_recommender import cli
from arxiv_recommender.recommendation.types import RecommendationItem, RecommendationRunResult
from arxiv_recommender.schemas import AppConfig, Paper, VectorizerConfig


class TestCli(unittest.TestCase):
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
            log_level="INFO",
        )
        self.result = RecommendationRunResult(
            recommendations=[
                RecommendationItem(
                    paper=Paper(
                        arxiv_id="1234.56789",
                        url="http://arxiv.org/abs/1234.56789",
                        title="Paper 1",
                        abstract="Abstract 1",
                    ),
                    score=0.9,
                ),
                RecommendationItem(
                    paper=Paper(title="Paper 2", abstract="Abstract 2"),
                    score=0.8,
                ),
            ],
            metrics_summary={"cache": {"hits": 1}},
            favorite_papers_count=2,
            candidate_papers_count=3,
        )

    @patch("arxiv_recommender.cli.argparse.ArgumentParser.parse_args")
    @patch("arxiv_recommender.cli.load_config")
    @patch("arxiv_recommender.cli.RecommendationPipeline")
    @patch("arxiv_recommender.cli.FileFavoritePapersProvider")
    @patch("arxiv_recommender.cli.MetricsCollector")
    @patch("arxiv_recommender.cli.load_vectorization_model")
    @patch("arxiv_recommender.cli.ArxivFetcher")
    @patch("arxiv_recommender.cli.setup_logging")
    def test_main_delegates_to_pipeline(
        self,
        mock_setup_logging: MagicMock,
        mock_fetcher_class: MagicMock,
        mock_load_vectorization_model: MagicMock,
        mock_metrics_class: MagicMock,
        mock_provider_class: MagicMock,
        mock_pipeline_class: MagicMock,
        mock_load_config: MagicMock,
        mock_parse_args: MagicMock,
    ) -> None:
        mock_parse_args.return_value = SimpleNamespace(
            config="config.json",
            date_of_pulling_papers="20260522",
            log_level=None,
            stats=False,
        )
        mock_load_config.return_value = self.config
        mock_pipeline = mock_pipeline_class.return_value
        mock_pipeline.run.return_value = self.result

        cli.main()

        mock_setup_logging.assert_called_once_with(level="INFO", json_format=True)
        mock_load_vectorization_model.assert_called_once_with(
            module_name=self.config.vectorizer.module_name,
            class_name=self.config.vectorizer.class_name,
            model_name=self.config.vectorizer.model_name,
            cache_size=self.config.vectorizer.cache_size,
        )
        mock_provider_class.assert_called_once_with(
            favorite_papers_path=self.config.favorite_papers_path,
            fetcher=mock_fetcher_class.return_value,
        )
        mock_pipeline_class.assert_called_once_with(
            config=self.config,
            fetcher=mock_fetcher_class.return_value,
            vectorizer=mock_load_vectorization_model.return_value,
            metrics=mock_metrics_class.return_value,
            favorite_papers_provider=mock_provider_class.return_value,
        )
        mock_pipeline.run.assert_called_once_with(date_of_pulling_papers="20260522")

    @patch("arxiv_recommender.cli.argparse.ArgumentParser.parse_args")
    @patch("arxiv_recommender.cli.load_config")
    @patch("arxiv_recommender.cli.RecommendationPipeline")
    @patch("arxiv_recommender.cli.FileFavoritePapersProvider")
    @patch("arxiv_recommender.cli.MetricsCollector")
    @patch("arxiv_recommender.cli.load_vectorization_model")
    @patch("arxiv_recommender.cli.ArxivFetcher")
    @patch("arxiv_recommender.cli.setup_logging")
    @patch("arxiv_recommender.cli.logging.getLogger")
    def test_main_logs_metrics_when_stats_enabled(
        self,
        mock_get_logger: MagicMock,
        mock_setup_logging: MagicMock,
        mock_fetcher_class: MagicMock,
        mock_load_vectorization_model: MagicMock,
        mock_metrics_class: MagicMock,
        mock_provider_class: MagicMock,
        mock_pipeline_class: MagicMock,
        mock_load_config: MagicMock,
        mock_parse_args: MagicMock,
    ) -> None:
        mock_parse_args.return_value = SimpleNamespace(
            config="config.json",
            date_of_pulling_papers=None,
            log_level=None,
            stats=True,
        )
        mock_load_config.return_value = self.config
        mock_pipeline_class.return_value.run.return_value = self.result
        mock_logger = mock_get_logger.return_value

        cli.main()

        mock_setup_logging.assert_called_once_with(level="INFO", json_format=True)
        mock_logger.info.assert_any_call("Metrics summary: %s", self.result.metrics_summary)
        mock_logger.info.assert_any_call(
            "%d. %s (%s) [%s]",
            1,
            "Paper 1",
            "Abstract 1",
            "http://arxiv.org/abs/1234.56789",
        )
        mock_fetcher_class.assert_called_once()
        mock_load_vectorization_model.assert_called_once()
        mock_metrics_class.assert_called_once()
        mock_provider_class.assert_called_once()

    @patch("arxiv_recommender.cli.argparse.ArgumentParser.parse_args")
    @patch("arxiv_recommender.cli.load_config")
    @patch("arxiv_recommender.cli.RecommendationPipeline")
    @patch("arxiv_recommender.cli.FileFavoritePapersProvider")
    @patch("arxiv_recommender.cli.MetricsCollector")
    @patch("arxiv_recommender.cli.load_vectorization_model")
    @patch("arxiv_recommender.cli.ArxivFetcher")
    @patch("arxiv_recommender.cli.setup_logging")
    def test_main_uses_log_level_override(
        self,
        mock_setup_logging: MagicMock,
        mock_fetcher_class: MagicMock,
        mock_load_vectorization_model: MagicMock,
        mock_metrics_class: MagicMock,
        mock_provider_class: MagicMock,
        mock_pipeline_class: MagicMock,
        mock_load_config: MagicMock,
        mock_parse_args: MagicMock,
    ) -> None:
        mock_parse_args.return_value = SimpleNamespace(
            config="config.json",
            date_of_pulling_papers=None,
            log_level="DEBUG",
            stats=False,
        )
        mock_load_config.return_value = self.config
        mock_pipeline_class.return_value.run.return_value = self.result

        cli.main()

        mock_setup_logging.assert_called_once_with(level="DEBUG", json_format=True)
        mock_fetcher_class.assert_called_once()
        mock_load_vectorization_model.assert_called_once()
        mock_metrics_class.assert_called_once()
        mock_provider_class.assert_called_once()


if __name__ == "__main__":
    unittest.main()
