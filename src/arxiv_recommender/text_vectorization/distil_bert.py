from arxiv_recommender.text_vectorization.huggingface_embed import HuggingFaceEmbedding


class DistilBERTEmbedding(HuggingFaceEmbedding):
    """Backward-compatible DistilBERT embedder.

    Prefer :class:`HuggingFaceEmbedding` for new code. This class preserves
    existing imports and configs that reference ``DistilBERTEmbedding``.
    """

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        cache_size: int = 1000,
    ) -> None:
        """Initializes the DistilBERT-compatible HuggingFace embedder.

        Args:
            model_name: HuggingFace DistilBERT model path or identifier.
            cache_size: Maximum number of embeddings to cache.
        """
        super().__init__(model_name=model_name, cache_size=cache_size)
