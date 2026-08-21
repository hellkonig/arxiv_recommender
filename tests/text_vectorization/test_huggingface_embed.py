import unittest
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from arxiv_recommender.text_vectorization.huggingface_embed import HuggingFaceEmbedding
from arxiv_recommender.text_vectorization.text_policy import TitleAbstractTextPolicy

MODEL_REVISION = "a" * 40
OTHER_MODEL_REVISION = "b" * 40


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

        embedder = HuggingFaceEmbedding("test-model", model_revision=MODEL_REVISION)
        embedding = embedder.process("sample text")

        self.assertIsInstance(embedding, np.ndarray)
        np.testing.assert_array_equal(embedding, np.array([2.0, 4.0], dtype=np.float32))
        mock_tokenizer_from_pretrained.assert_called_once_with(
            "test-model", revision=MODEL_REVISION
        )
        mock_model_from_pretrained.assert_called_once_with("test-model", revision=MODEL_REVISION)
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

        embedder = HuggingFaceEmbedding("test-model", model_revision=MODEL_REVISION)
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

        embedder = HuggingFaceEmbedding("test-model", model_revision=MODEL_REVISION)

        self.assertEqual(embedder.pooling_strategy, "mean")
        self.assertFalse(embedder.normalize_embeddings)

    @patch("arxiv_recommender.text_vectorization.huggingface_embed.AutoModel.from_pretrained")
    @patch("arxiv_recommender.text_vectorization.huggingface_embed.AutoTokenizer.from_pretrained")
    def test_bge_small_auto_uses_profile_settings(
        self, mock_tokenizer_from_pretrained: Any, mock_model_from_pretrained: Any
    ) -> None:
        mock_tokenizer_from_pretrained.return_value = MagicMock()
        mock_model_from_pretrained.return_value = MagicMock()

        embedder = HuggingFaceEmbedding(
            "BAAI/bge-small-en-v1.5",
            model_revision=MODEL_REVISION,
        )

        self.assertEqual(embedder.pooling_strategy, "cls")
        self.assertTrue(embedder.normalize_embeddings)
        self.assertIn("pooling=cls", embedder.embedding_config_id)
        self.assertIn("normalize=True", embedder.embedding_config_id)

    @patch("arxiv_recommender.text_vectorization.huggingface_embed.AutoModel.from_pretrained")
    @patch("arxiv_recommender.text_vectorization.huggingface_embed.AutoTokenizer.from_pretrained")
    def test_provenance_uses_resolved_embedding_contract(
        self, mock_tokenizer_from_pretrained: Any, mock_model_from_pretrained: Any
    ) -> None:
        mock_tokenizer_from_pretrained.return_value = MagicMock()
        mock_model_from_pretrained.return_value = MagicMock()

        embedder = HuggingFaceEmbedding(
            "BAAI/bge-small-en-v1.5",
            pooling_strategy="auto",
            normalize_embeddings="auto",
            max_length=256,
            model_revision=MODEL_REVISION,
        )

        self.assertEqual(
            embedder.provenance.model_dump(),
            {
                "name": "BAAI/bge-small-en-v1.5",
                "version": MODEL_REVISION,
                "config": {
                    "implementation": {
                        "name": "huggingface_embedding",
                        "version": "1.0.0",
                    },
                    "pooling_strategy": "cls",
                    "normalize_embeddings": True,
                    "max_length": 256,
                    "text_policy": {
                        "name": "title_abstract",
                        "version": "1.0.0",
                        "config": {"separator": " "},
                    },
                },
            },
        )

    @patch("arxiv_recommender.text_vectorization.huggingface_embed.AutoModel.from_pretrained")
    @patch("arxiv_recommender.text_vectorization.huggingface_embed.AutoTokenizer.from_pretrained")
    def test_provenance_uses_injected_text_policy(
        self, mock_tokenizer_from_pretrained: Any, mock_model_from_pretrained: Any
    ) -> None:
        mock_tokenizer_from_pretrained.return_value = MagicMock()
        mock_model_from_pretrained.return_value = MagicMock()
        policy = TitleAbstractTextPolicy(separator="\n")

        embedder = HuggingFaceEmbedding(
            "test-model",
            text_policy=policy,
            model_revision=MODEL_REVISION,
        )

        self.assertIs(embedder.text_policy, policy)
        self.assertEqual(
            embedder.provenance.config["text_policy"],
            {
                "name": "title_abstract",
                "version": "1.0.0",
                "config": {"separator": "\n"},
            },
        )

    def test_bge_small_rejects_incorrect_explicit_pooling(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "pooling_strategy must be 'cls', got 'mean'",
        ):
            HuggingFaceEmbedding(
                "BAAI/bge-small-en-v1.5",
                pooling_strategy="mean",
                normalize_embeddings=True,
                model_revision=MODEL_REVISION,
            )

    def test_bge_small_rejects_incorrect_explicit_normalization(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "normalize_embeddings must be True, got False",
        ):
            HuggingFaceEmbedding(
                "BAAI/bge-small-en-v1.5",
                pooling_strategy="cls",
                normalize_embeddings=False,
                model_revision=MODEL_REVISION,
            )

    def test_rejects_moving_or_noncanonical_model_revision(self) -> None:
        for invalid_revision in ("main", "v1.0", "A" * 40, "a" * 39):
            with self.subTest(model_revision=invalid_revision):
                with self.assertRaisesRegex(
                    ValueError,
                    "full 40-character lowercase commit SHA",
                ):
                    HuggingFaceEmbedding(
                        "test-model",
                        model_revision=invalid_revision,
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
            model_revision=MODEL_REVISION,
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

        mean_embedder = HuggingFaceEmbedding(
            "test-model",
            pooling_strategy="mean",
            model_revision=MODEL_REVISION,
        )
        cls_embedder = HuggingFaceEmbedding(
            "test-model",
            pooling_strategy="cls",
            model_revision=MODEL_REVISION,
        )

        self.assertNotEqual(mean_embedder.embedding_config_id, cls_embedder.embedding_config_id)
        self.assertNotEqual(
            mean_embedder.cache._compute_key("same text"),
            cls_embedder.cache._compute_key("same text"),
        )

    @patch("arxiv_recommender.text_vectorization.huggingface_embed.AutoModel.from_pretrained")
    @patch("arxiv_recommender.text_vectorization.huggingface_embed.AutoTokenizer.from_pretrained")
    def test_model_revision_versions_cache_namespace(
        self, mock_tokenizer_from_pretrained: Any, mock_model_from_pretrained: Any
    ) -> None:
        mock_tokenizer_from_pretrained.return_value = MagicMock()
        mock_model_from_pretrained.return_value = MagicMock()

        first_embedder = HuggingFaceEmbedding(
            "test-model",
            model_revision=MODEL_REVISION,
        )
        second_embedder = HuggingFaceEmbedding(
            "test-model",
            model_revision=OTHER_MODEL_REVISION,
        )

        self.assertNotEqual(
            first_embedder.cache._compute_key("same text"),
            second_embedder.cache._compute_key("same text"),
        )


if __name__ == "__main__":
    unittest.main()
