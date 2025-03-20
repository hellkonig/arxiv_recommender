import unittest
from unittest.mock import patch
from arxiv_recommender.utils.model_loader import load_vectorization_model


class TestModelLoader(unittest.TestCase):
    """Test cases for dynamic text vectorizer loading."""

    @patch("importlib.import_module")
    def test_load_default_vectorizer(self, mock_import):
        """
        Test loading the default DistilBERTEmbedding vectorizer.
        """
        # Mock module and class
        mock_module = unittest.mock.MagicMock()
        mock_class = unittest.mock.MagicMock()
        mock_module.DistilBERTEmbedding = mock_class
        mock_import.return_value = mock_module

        # Load default vectorizer
        vectorizer = load_vectorization_model()

        # Ensure the correct module was loaded
        mock_import.assert_called_once_with(
            "arxiv_recommender.text_vectorization.distil_bert"
        )
        # Ensure the correct class was instantiated
        mock_class.assert_called_once()

        # Ensure instance is returned
        self.assertEqual(vectorizer, mock_class())

    @patch("importlib.import_module")
    def test_load_custom_vectorizer(self, mock_import):
        """
        Test loading a custom text vectorization model dynamically.
        """
        mock_module = unittest.mock.MagicMock()
        mock_class = unittest.mock.MagicMock()
        mock_module.CustomVectorizer = mock_class
        mock_import.return_value = mock_module

        # Load a custom vectorizer
        vectorizer = load_vectorization_model(
            "custom_vectorizer",
            "CustomVectorizer"
        )

        # Ensure the correct module was loaded
        mock_import.assert_called_once_with(
            "arxiv_recommender.text_vectorization.custom_vectorizer"
        )
        # Ensure the correct class was instantiated
        mock_class.assert_called_once()

        # Ensure instance is returned
        self.assertEqual(vectorizer, mock_class())

    def test_load_invalid_vectorizer(self):
        """
        Test behavior when an invalid vectorizer is specified.
        Expect ImportError to be raised.
        """
        with self.assertRaises(ImportError):
            load_vectorization_model(
                "non_existent_model",
                "NonExistentModel"
            )


if __name__ == "__main__":
    unittest.main()
