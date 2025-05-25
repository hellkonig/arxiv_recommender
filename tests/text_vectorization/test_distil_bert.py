import unittest
from unittest.mock import patch, MagicMock
import torch

from arxiv_recommender.text_vectorization.distil_bert import DistilBERTEmbedding


class TestDistilBERTEmbedding(unittest.TestCase):

    @patch("arxiv_recommender.text_vectorization.distil_bert.DistilBertModel.from_pretrained")
    @patch("arxiv_recommender.text_vectorization.distil_bert.DistilBertTokenizer.from_pretrained")
    def test_process_returns_expected_shape(self, mock_tokenizer_from_pretrained, mock_model_from_pretrained):
        # Mock the tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.randint(0, 1000, (1, 512)),
            "attention_mask": torch.ones(1, 512),
        }
        mock_tokenizer_from_pretrained.return_value = mock_tokenizer
        # Mock the model
        mock_model = MagicMock()
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.rand(1, 512, 768)  # (batch_size, seq_len, hidden_dim)
        mock_model.return_value = mock_output
        mock_model_from_pretrained.return_value = mock_model

        # Instantiate and run
        embedder = DistilBERTEmbedding("distilbert-base-uncased")
        embedding = embedder.process("This is a sample input text.")

        # Assertions
        self.assertIsInstance(embedding, torch.Tensor)
        self.assertEqual(embedding.shape, (768,))  # After mean pooling

        # Ensure model/tokenizer were called correctly
        mock_tokenizer_from_pretrained.assert_called_once()
        mock_model_from_pretrained.assert_called_once()

if __name__ == "__main__":
    unittest.main()
