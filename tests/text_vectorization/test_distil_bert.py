import unittest
from typing import Any
from unittest.mock import patch

from arxiv_recommender.text_vectorization.distil_bert import DistilBERTEmbedding
from arxiv_recommender.text_vectorization.huggingface_embed import HuggingFaceEmbedding
from arxiv_recommender.text_vectorization import TextEmbedder

MODEL_REVISION = "a" * 40


class TestDistilBERTEmbedding(unittest.TestCase):
    @patch("arxiv_recommender.text_vectorization.huggingface_embed.AutoModel.from_pretrained")
    @patch("arxiv_recommender.text_vectorization.huggingface_embed.AutoTokenizer.from_pretrained")
    def test_distilbert_embedding_is_backward_compatible_wrapper(
        self, mock_tokenizer_from_pretrained: Any, mock_model_from_pretrained: Any
    ) -> None:
        embedder = DistilBERTEmbedding(model_revision=MODEL_REVISION)

        self.assertIsInstance(embedder, TextEmbedder)
        self.assertIsInstance(embedder, HuggingFaceEmbedding)
        self.assertEqual(embedder.model_name, "distilbert-base-uncased")
        mock_tokenizer_from_pretrained.assert_called_once_with(
            "distilbert-base-uncased", revision=MODEL_REVISION
        )
        mock_model_from_pretrained.assert_called_once_with(
            "distilbert-base-uncased", revision=MODEL_REVISION
        )


if __name__ == "__main__":
    unittest.main()
