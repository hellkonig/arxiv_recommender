import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from arxiv_recommender import cli
from arxiv_recommender.recommendation.types import RecommendationRunResult
from arxiv_recommender.schemas import AppConfig, VectorizerConfig


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
                {"title": "Paper 1", "abstract": "Abstract 1", "score": 0.9},
                {"title": "Paper 2", "abstract": "Abstract 2", "score": 0.8},
            ],
            metrics_summary={"cache": {"hits": 1}},
            favorite_papers_count=2,
            candidate_papers_count=3,
        )

    @patch("arxiv_recommender.cli.argparse.ArgumentParser.parse_args")
    @patch("arxiv_recommender.cli.load_config")
    @patch("arxiv_recommender.cli.RecommendationPipeline")
    @patch("arxiv_recommender.cli.setup_logging")
    def test_main_delegates_to_pipeline(
        self,
        mock_setup_logging: MagicMock,
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
        mock_pipeline_class.assert_called_once_with(config=self.config)
        mock_pipeline.run.assert_called_once_with(date_of_pulling_papers="20260522")

    @patch("arxiv_recommender.cli.argparse.ArgumentParser.parse_args")
    @patch("arxiv_recommender.cli.load_config")
    @patch("arxiv_recommender.cli.RecommendationPipeline")
    @patch("arxiv_recommender.cli.setup_logging")
    @patch("arxiv_recommender.cli.logging.getLogger")
    def test_main_logs_metrics_when_stats_enabled(
        self,
        mock_get_logger: MagicMock,
        mock_setup_logging: MagicMock,
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

    @patch("arxiv_recommender.cli.argparse.ArgumentParser.parse_args")
    @patch("arxiv_recommender.cli.load_config")
    @patch("arxiv_recommender.cli.RecommendationPipeline")
    @patch("arxiv_recommender.cli.setup_logging")
    def test_main_uses_log_level_override(
        self,
        mock_setup_logging: MagicMock,
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


if __name__ == "__main__":
    unittest.main()
