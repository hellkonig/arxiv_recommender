from typing import Any, cast

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from arxiv_recommender.embedding_config import (
    AutoSetting,
    NormalizeEmbeddingsSetting,
    PoolingStrategy,
    resolve_embedding_config,
)
from arxiv_recommender.text_vectorization.base import TextEmbedder
from arxiv_recommender.text_vectorization.cache import EmbeddingCache
from arxiv_recommender.text_vectorization.pooling import create_pooler


class HuggingFaceEmbedding(TextEmbedder):
    """Generic HuggingFace text embedder with model-aware embedding settings."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        cache_size: int = 1000,
        pooling_strategy: PoolingStrategy | str = PoolingStrategy.AUTO,
        normalize_embeddings: NormalizeEmbeddingsSetting | str = AutoSetting.AUTO,
        max_length: int = 512,
    ) -> None:
        """Initializes tokenizer, model, and embedding cache.

        Args:
            model_name: HuggingFace model path or identifier.
            cache_size: Maximum number of embeddings to cache.
            pooling_strategy: Token pooling strategy, or auto for known model profiles.
            normalize_embeddings: Whether to L2-normalize embeddings, or auto for known profiles.
            max_length: Maximum token length for truncation.
        """
        self.model_name = model_name
        self.embedding_config = resolve_embedding_config(
            model_name=model_name,
            pooling_strategy=pooling_strategy,
            normalize_embeddings=normalize_embeddings,
        )
        self.pooling_strategy = self.embedding_config.pooling_strategy.value
        self.normalize_embeddings = self.embedding_config.normalize_embeddings
        self._pooler = create_pooler(self.embedding_config.pooling_strategy)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.max_length = max_length
        self.embedding_config_id = self.embedding_config.cache_namespace(model_name, max_length)
        self.cache = EmbeddingCache(max_size=cache_size, namespace=self.embedding_config_id)

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
        """Get embedding vectors for tokenized text.

        Args:
            tokenized_text: Tokenized HuggingFace model inputs.

        Returns:
            Text embedding as tensor.
        """
        with torch.no_grad():
            outputs = self.model(**tokenized_text)

        embeddings = cast(torch.Tensor, outputs.last_hidden_state)
        attention_mask = tokenized_text["attention_mask"]
        pooled_embeddings = self._pooler(embeddings, attention_mask)

        if self.normalize_embeddings:
            pooled_embeddings = torch.nn.functional.normalize(pooled_embeddings, p=2, dim=1)

        return pooled_embeddings[0]

    def get_cache_stats(self) -> dict[str, Any]:
        """Get cache performance statistics.

        Returns:
            Dictionary with cache hits, misses, and hit rate.
        """
        return self.cache.stats()
