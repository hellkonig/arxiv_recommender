import unittest
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import patch

import numpy as np

from arxiv_recommender.provenance import ModelProvenance
from arxiv_recommender.text_vectorization import TextEmbedder
from arxiv_recommender.utils.model_loader import (
    VectorizationModelLoadError,
    load_vectorization_model,
)


class FakeEmbedder(TextEmbedder):
    def __init__(self, model_name: str, cache_size: int, **kwargs: Any) -> None:
        self.model_name = model_name
        self.cache_size = cache_size
        self.kwargs = kwargs

    @property
    def provenance(self) -> ModelProvenance:
        return ModelProvenance(name=self.model_name, version="1.0.0")

    def process(self, text: str) -> np.ndarray:
        return np.array([len(text)])

    def get_cache_stats(self) -> dict[str, int]:
        return {"hits": 0, "misses": 0}


class BrokenEmbedder(TextEmbedder):
    def __init__(self, model_name: str, cache_size: int) -> None:
        raise ValueError("model cannot be loaded")

    @property
    def provenance(self) -> ModelProvenance:
        return ModelProvenance(name="broken", version="1.0.0")

    def process(self, text: str) -> np.ndarray:
        return np.array([len(text)])

    def get_cache_stats(self) -> dict[str, int]:
        return {"hits": 0, "misses": 0}


class NotEmbedder:
    def __init__(self, model_name: str, cache_size: int) -> None:
        self.model_name = model_name
        self.cache_size = cache_size


class TestModelLoader(unittest.TestCase):
    """Test cases for dynamic text vectorizer loading."""

    @patch("importlib.import_module")
    def test_load_default_vectorizer(self, mock_import: Any) -> None:
        """
        Test loading the default DistilBERTEmbedding vectorizer.
        """
        mock_module = SimpleNamespace(DistilBERTEmbedding=FakeEmbedder)
        mock_import.return_value = mock_module

        vectorizer = load_vectorization_model(
            "distil_bert", "DistilBERTEmbedding", "distilbert-base-uncased", 1000
        )

        mock_import.assert_called_once_with("arxiv_recommender.text_vectorization.distil_bert")
        self.assertIsInstance(vectorizer, FakeEmbedder)
        fake_vectorizer = cast(FakeEmbedder, vectorizer)
        self.assertEqual(fake_vectorizer.model_name, "distilbert-base-uncased")
        self.assertEqual(fake_vectorizer.cache_size, 1000)

    @patch("importlib.import_module")
    def test_load_custom_vectorizer(self, mock_import: Any) -> None:
        """
        Test loading a custom text vectorization model dynamically.
        """
        mock_module = SimpleNamespace(CustomVectorizer=FakeEmbedder)
        mock_import.return_value = mock_module

        vectorizer = load_vectorization_model(
            "custom_vectorizer", "CustomVectorizer", "custom_vectorinzer_model_name", 1000
        )

        mock_import.assert_called_once_with(
            "arxiv_recommender.text_vectorization.custom_vectorizer"
        )
        self.assertIsInstance(vectorizer, FakeEmbedder)
        fake_vectorizer = cast(FakeEmbedder, vectorizer)
        self.assertEqual(fake_vectorizer.model_name, "custom_vectorinzer_model_name")
        self.assertEqual(fake_vectorizer.cache_size, 1000)

    @patch("importlib.import_module")
    def test_load_vectorizer_passes_options(self, mock_import: Any) -> None:
        """
        Test passing optional vectorizer settings to the configured class.
        """
        mock_module = SimpleNamespace(CustomVectorizer=FakeEmbedder)
        mock_import.return_value = mock_module

        vectorizer = load_vectorization_model(
            "custom_vectorizer",
            "CustomVectorizer",
            "custom_model",
            1000,
            vectorizer_options={
                "pooling_strategy": "cls",
                "normalize_embeddings": True,
                "max_length": 256,
            },
        )

        fake_vectorizer = cast(FakeEmbedder, vectorizer)
        self.assertEqual(
            fake_vectorizer.kwargs,
            {
                "pooling_strategy": "cls",
                "normalize_embeddings": True,
                "max_length": 256,
            },
        )

    @patch("importlib.import_module")
    def test_load_missing_vectorizer_module(self, mock_import: Any) -> None:
        full_module_name = "arxiv_recommender.text_vectorization.non_existent_module"
        mock_import.side_effect = ModuleNotFoundError(
            f"No module named '{full_module_name}'",
            name=full_module_name,
        )

        with self.assertRaisesRegex(VectorizationModelLoadError, "module .* was not found"):
            load_vectorization_model(
                "non_existent_module", "NonExistentModel", "non_existent_model", 1000
            )

    @patch("importlib.import_module")
    def test_load_reports_missing_vectorizer_dependency(self, mock_import: Any) -> None:
        mock_import.side_effect = ModuleNotFoundError(
            "No module named 'transformers'",
            name="transformers",
        )

        with self.assertRaisesRegex(VectorizationModelLoadError, "dependency 'transformers'"):
            load_vectorization_model("broken_module", "BrokenVectorizer", "broken-model", 1000)

    @patch("importlib.import_module")
    def test_load_missing_vectorizer_class(self, mock_import: Any) -> None:
        mock_import.return_value = SimpleNamespace()

        with self.assertRaisesRegex(VectorizationModelLoadError, "class 'MissingEmbedder'"):
            load_vectorization_model("custom_vectorizer", "MissingEmbedder", "model-name", 1000)

    @patch("importlib.import_module")
    def test_load_rejects_non_text_embedder(self, mock_import: Any) -> None:
        mock_import.return_value = SimpleNamespace(NotEmbedder=NotEmbedder)

        with self.assertRaisesRegex(VectorizationModelLoadError, "must inherit from TextEmbedder"):
            load_vectorization_model("custom_vectorizer", "NotEmbedder", "model-name", 1000)

    @patch("importlib.import_module")
    def test_load_reports_model_instantiation_failure(self, mock_import: Any) -> None:
        mock_import.return_value = SimpleNamespace(BrokenEmbedder=BrokenEmbedder)

        with self.assertRaisesRegex(
            VectorizationModelLoadError,
            "Failed to instantiate vectorizer .* with model 'bad-model'",
        ):
            load_vectorization_model("custom_vectorizer", "BrokenEmbedder", "bad-model", 1000)


if __name__ == "__main__":
    unittest.main()
