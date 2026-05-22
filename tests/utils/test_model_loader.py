import unittest
from typing import Any
from unittest.mock import patch

import numpy as np

from arxiv_recommender.text_vectorization import TextEmbedder
from arxiv_recommender.utils.model_loader import load_vectorization_model


class FakeEmbedder(TextEmbedder):
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
        # Mock module and class
        mock_module = unittest.mock.MagicMock()
        mock_class = unittest.mock.MagicMock()
        fake_embedder = FakeEmbedder()
        mock_class.return_value = fake_embedder
        mock_module.DistilBERTEmbedding = mock_class
        mock_import.return_value = mock_module

        # Load default vectorizer
        vectorizer = load_vectorization_model(
            "distil_bert", "DistilBERTEmbedding", "distilbert-base-uncased", 1000
        )

        # Ensure the correct module was loaded
        mock_import.assert_called_once_with("arxiv_recommender.text_vectorization.distil_bert")
        # Ensure the correct class was instantiated with cache_size
        mock_class.assert_called_once_with("distilbert-base-uncased", cache_size=1000)

        # Ensure instance is returned
        self.assertEqual(vectorizer, fake_embedder)

    @patch("importlib.import_module")
    def test_load_custom_vectorizer(self, mock_import: Any) -> None:
        """
        Test loading a custom text vectorization model dynamically.
        """
        mock_module = unittest.mock.MagicMock()
        mock_class = unittest.mock.MagicMock()
        fake_embedder = FakeEmbedder()
        mock_class.return_value = fake_embedder
        mock_module.CustomVectorizer = mock_class
        mock_import.return_value = mock_module

        # Load a custom vectorizer
        vectorizer = load_vectorization_model(
            "custom_vectorizer", "CustomVectorizer", "custom_vectorinzer_model_name", 1000
        )

        # Ensure the correct module was loaded
        mock_import.assert_called_once_with(
            "arxiv_recommender.text_vectorization.custom_vectorizer"
        )
        # Ensure the correct class was instantiated with cache_size
        mock_class.assert_called_once_with("custom_vectorinzer_model_name", cache_size=1000)

        # Ensure instance is returned
        self.assertEqual(vectorizer, fake_embedder)

    def test_load_invalid_vectorizer(self) -> None:
        """
        Test behavior when an invalid vectorizer is specified.
        Expect ImportError to be raised.
        """
        with self.assertRaises(ImportError):
            load_vectorization_model(
                "non_existent_module", "NonExistentModel", "non_existent_model", 1000
            )

    @patch("importlib.import_module")
    def test_load_rejects_non_text_embedder(self, mock_import: Any) -> None:
        mock_module = unittest.mock.MagicMock()
        mock_module.NotEmbedder = NotEmbedder
        mock_import.return_value = mock_module

        with self.assertRaises(ImportError):
            load_vectorization_model("custom_vectorizer", "NotEmbedder", "model-name", 1000)


if __name__ == "__main__":
    unittest.main()
