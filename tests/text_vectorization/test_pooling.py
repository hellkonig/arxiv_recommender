import pytest
import torch

from arxiv_recommender.embedding_config import PoolingStrategy
from arxiv_recommender.text_vectorization.pooling import create_pooler


def test_create_pooler_returns_cls_pooler() -> None:
    embeddings = torch.tensor([[[3.0, 4.0], [100.0, 100.0]]])
    attention_mask = torch.tensor([[1, 1]])

    pooler = create_pooler(PoolingStrategy.CLS)

    torch.testing.assert_close(pooler(embeddings, attention_mask), torch.tensor([[3.0, 4.0]]))


def test_create_pooler_returns_attention_aware_mean_pooler() -> None:
    embeddings = torch.tensor([[[1.0, 3.0], [3.0, 5.0], [100.0, 100.0]]])
    attention_mask = torch.tensor([[1, 1, 0]])

    pooler = create_pooler(PoolingStrategy.MEAN)

    torch.testing.assert_close(pooler(embeddings, attention_mask), torch.tensor([[2.0, 4.0]]))


def test_create_pooler_rejects_auto_strategy() -> None:
    with pytest.raises(ValueError, match="Unsupported pooling strategy: auto"):
        create_pooler(PoolingStrategy.AUTO)
