import json
import os
import tempfile
import unittest

from pydantic import ValidationError

from arxiv_recommender.schemas import AppConfig
from arxiv_recommender.utils.config_loader import load_config


class TestConfigLoader(unittest.TestCase):
    def _write_config(self, config_data: dict[str, object]) -> str:
        temp_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)

        def remove_temp_file() -> None:
            if os.path.exists(temp_file.name):
                os.remove(temp_file.name)

        self.addCleanup(remove_temp_file)
        with temp_file:
            json.dump(config_data, temp_file)
        return temp_file.name

    def test_load_config_returns_validated_app_config(self) -> None:
        config_path = self._write_config(
            {
                "favorite_papers_path": "favorite_papers.json",
                "vectorizer": {
                    "module_name": "huggingface_embed",
                    "class_name": "HuggingFaceEmbedding",
                    "model_name": "distilbert-base-uncased",
                    "model_revision": "a" * 40,
                    "cache_size": 1000,
                },
                "top_k": 5,
                "log_level": "debug",
            }
        )

        config = load_config(config_path)

        self.assertIsInstance(config, AppConfig)
        self.assertEqual(config.favorite_papers_path, "favorite_papers.json")
        self.assertEqual(config.top_k, 5)
        self.assertEqual(config.log_level, "DEBUG")

    def test_load_config_raises_for_missing_file(self) -> None:
        with self.assertRaisesRegex(FileNotFoundError, "Configuration file not found"):
            load_config("missing_config.json")

    def test_load_config_raises_for_invalid_config_shape(self) -> None:
        config_path = self._write_config({"favorite_papers_path": "favorite_papers.json"})

        with self.assertRaises(ValidationError):
            load_config(config_path)


if __name__ == "__main__":
    unittest.main()
