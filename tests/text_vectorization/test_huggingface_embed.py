import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from arxiv_recommender.text_vectorization.huggingface_embed import HuggingFaceEmbedding


class TestHuggingFaceEmbedding(unittest.TestCase):
    @patch("arxiv_recommender.text_vectorization.huggingface_embed.AutoModel.from_pretrained")
    @patch("arxiv_recommender.text_vectorization.huggingface_embed.AutoTokenizer.from_pretrained")
    def test_process_returns_attention_masked_mean_embedding(
        self, mock_tokenizer_from_pretrained: Any, mock_model_from_pretrained: Any
    ) -> None:
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 102, 0]]),
            "attention_mask": torch.tensor([[1, 1, 0]]),
        }
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
        mock_model.return_value = mock_output
        mock_model_from_pretrained.return_value = mock_model

        embedder = HuggingFaceEmbedding("test-model")
        embedding = embedder.process("sample text")

        self.assertIsInstance(embedding, np.ndarray)
        np.testing.assert_array_equal(embedding, np.array([2.0, 4.0], dtype=np.float32))
        mock_tokenizer_from_pretrained.assert_called_once_with("test-model")
        mock_model_from_pretrained.assert_called_once_with("test-model")
        mock_model.eval.assert_called_once()

    @patch("arxiv_recommender.text_vectorization.huggingface_embed.AutoModel.from_pretrained")
    @patch("arxiv_recommender.text_vectorization.huggingface_embed.AutoTokenizer.from_pretrained")
    def test_process_uses_cache(
        self, mock_tokenizer_from_pretrained: Any, mock_model_from_pretrained: Any
    ) -> None:
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101]]),
            "attention_mask": torch.tensor([[1]]),
        }
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.tensor([[[1.0, 2.0]]])
        mock_model.return_value = mock_output
        mock_model_from_pretrained.return_value = mock_model

        embedder = HuggingFaceEmbedding("test-model")
        first_embedding = embedder.process("same text")
        second_embedding = embedder.process("same text")

        np.testing.assert_array_equal(first_embedding, second_embedding)
        mock_model.assert_called_once()
        self.assertEqual(embedder.get_cache_stats()["hits"], 1)

    @patch("arxiv_recommender.text_vectorization.huggingface_embed.AutoModel.from_pretrained")
    @patch("arxiv_recommender.text_vectorization.huggingface_embed.AutoTokenizer.from_pretrained")
    def test_unknown_model_auto_uses_mean_pooling_without_normalization(
        self, mock_tokenizer_from_pretrained: Any, mock_model_from_pretrained: Any
    ) -> None:
        mock_tokenizer_from_pretrained.return_value = MagicMock()
        mock_model_from_pretrained.return_value = MagicMock()

        embedder = HuggingFaceEmbedding("test-model")

        self.assertEqual(embedder.pooling_strategy, "mean")
        self.assertFalse(embedder.normalize_embeddings)

    @patch("arxiv_recommender.text_vectorization.huggingface_embed.AutoModel.from_pretrained")
    @patch("arxiv_recommender.text_vectorization.huggingface_embed.AutoTokenizer.from_pretrained")
    def test_bge_small_auto_uses_profile_settings(
        self, mock_tokenizer_from_pretrained: Any, mock_model_from_pretrained: Any
    ) -> None:
        mock_tokenizer_from_pretrained.return_value = MagicMock()
        mock_model_from_pretrained.return_value = MagicMock()

        embedder = HuggingFaceEmbedding("BAAI/bge-small-en-v1.5")

        self.assertEqual(embedder.pooling_strategy, "cls")
        self.assertTrue(embedder.normalize_embeddings)
        self.assertIn("pooling=cls", embedder.embedding_config_id)
        self.assertIn("normalize=True", embedder.embedding_config_id)

    def test_bge_small_rejects_incorrect_explicit_pooling(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "BAAI/bge-small-en-v1.5 requires pooling_strategy='cls'",
        ):
            HuggingFaceEmbedding(
                "BAAI/bge-small-en-v1.5",
                pooling_strategy="mean",
                normalize_embeddings=True,
            )

    def test_bge_small_rejects_incorrect_explicit_normalization(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "BAAI/bge-small-en-v1.5 requires pooling_strategy='cls'",
        ):
            HuggingFaceEmbedding(
                "BAAI/bge-small-en-v1.5",
                pooling_strategy="cls",
                normalize_embeddings=False,
            )

    @patch("arxiv_recommender.text_vectorization.huggingface_embed.AutoModel.from_pretrained")
    @patch("arxiv_recommender.text_vectorization.huggingface_embed.AutoTokenizer.from_pretrained")
    def test_process_can_return_normalized_cls_embedding(
        self, mock_tokenizer_from_pretrained: Any, mock_model_from_pretrained: Any
    ) -> None:
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[101, 102]]),
            "attention_mask": torch.tensor([[1, 1]]),
        }
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.tensor([[[3.0, 4.0], [100.0, 100.0]]])
        mock_model.return_value = mock_output
        mock_model_from_pretrained.return_value = mock_model

        embedder = HuggingFaceEmbedding(
            "test-model",
            pooling_strategy="cls",
            normalize_embeddings=True,
        )
        embedding = embedder.process("sample text")

        np.testing.assert_allclose(embedding, np.array([0.6, 0.8], dtype=np.float32))
        self.assertAlmostEqual(float(np.linalg.norm(embedding)), 1.0)

    @patch("arxiv_recommender.text_vectorization.huggingface_embed.AutoModel.from_pretrained")
    @patch("arxiv_recommender.text_vectorization.huggingface_embed.AutoTokenizer.from_pretrained")
    def test_embedding_config_versions_cache_namespace(
        self, mock_tokenizer_from_pretrained: Any, mock_model_from_pretrained: Any
    ) -> None:
        mock_tokenizer_from_pretrained.return_value = MagicMock()
        mock_model_from_pretrained.return_value = MagicMock()

        mean_embedder = HuggingFaceEmbedding("test-model", pooling_strategy="mean")
        cls_embedder = HuggingFaceEmbedding("test-model", pooling_strategy="cls")

        self.assertNotEqual(mean_embedder.embedding_config_id, cls_embedder.embedding_config_id)
        self.assertNotEqual(
            mean_embedder.cache._compute_key("same text"),
            cls_embedder.cache._compute_key("same text"),
        )


if __name__ == "__main__":
    unittest.main()
