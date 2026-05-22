from typing import Any, cast

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from arxiv_recommender.text_vectorization.base import TextEmbedder
from arxiv_recommender.text_vectorization.cache import EmbeddingCache


class HuggingFaceEmbedding(TextEmbedder):
    """Generic HuggingFace text embedder using mean pooled token embeddings."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        cache_size: int = 1000,
    ) -> None:
        """Initializes tokenizer, model, and embedding cache.

        Args:
            model_name: HuggingFace model path or identifier.
            cache_size: Maximum number of embeddings to cache.
        """
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.max_length = 512
        self.cache = EmbeddingCache(max_size=cache_size)

    def process(self, text: str) -> np.ndarray:
        """Generate an embedding vector for the given text.

        Args:
            text: Input text string.

        Returns:
            A numpy array representing the text embedding.
        """
        cached_embedding = self.cache.get(text)
        if cached_embedding is not None:
            return cached_embedding

        tokenized_text = self.tokenize(text)
        embedding = self.vectorize(tokenized_text)
        result = embedding.detach().cpu().numpy()
        self.cache.put(text, result)

        return result

    def tokenize(self, text: str) -> dict[str, torch.Tensor]:
        """Tokenizes input text for the configured HuggingFace model.

        Args:
            text: Input text string.

        Returns:
            Tokenized tensors keyed by HuggingFace input name.
        """
        tokenized_text = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.max_length,
        )
        return cast(dict[str, torch.Tensor], tokenized_text)

    def vectorize(self, tokenized_text: dict[str, torch.Tensor]) -> torch.Tensor:
        """Get embedding vectors for tokenized text with attention-aware mean pooling.

        Args:
            tokenized_text: Tokenized HuggingFace model inputs.

        Returns:
            Mean pooled text embedding as tensor.
        """
        with torch.no_grad():
            outputs = self.model(**tokenized_text)

        embeddings = outputs.last_hidden_state
        attention_mask = tokenized_text["attention_mask"]
        return self._mean_pool(embeddings, attention_mask)

    def _mean_pool(self, embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
        masked_embeddings = embeddings * mask
        summed_embeddings = torch.sum(masked_embeddings, dim=1)
        token_counts = torch.clamp(mask.sum(dim=1), min=1e-9)
        sentence_embeddings = summed_embeddings / token_counts
        return torch.mean(sentence_embeddings, dim=0, keepdim=False)

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Dictionary with cache hits, misses, and hit rate.
        """
        return self.cache.stats()
