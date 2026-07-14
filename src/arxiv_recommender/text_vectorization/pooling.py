from collections.abc import Callable

import torch

from arxiv_recommender.text_vectorization.config import PoolingStrategy

Pooler = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


def create_pooler(pooling_strategy: PoolingStrategy) -> Pooler:
    """Create a token pooler for HuggingFace hidden states."""
    if pooling_strategy is PoolingStrategy.CLS:
        return cls_pool
    if pooling_strategy is PoolingStrategy.MEAN:
        return mean_pool

    raise ValueError(f"Unsupported pooling strategy: {pooling_strategy.value}")


def cls_pool(embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Use the first token hidden state as the text embedding."""
    del attention_mask
    return embeddings[:, 0]


def mean_pool(embeddings: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Average non-padding token hidden states."""
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed_embeddings = torch.sum(masked_embeddings, dim=1)
    token_counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed_embeddings / token_counts
