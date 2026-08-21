import pytest

from arxiv_recommender.text_vectorization.config import (
    AutoSetting,
    PoolingStrategy,
    resolve_embedding_config,
)


def test_unknown_model_auto_uses_default_embedding_config() -> None:
    config = resolve_embedding_config("unknown-model")

    assert config.pooling_strategy is PoolingStrategy.MEAN
    assert config.normalize_embeddings is False


def test_unknown_model_allows_explicit_embedding_config() -> None:
    config = resolve_embedding_config(
        "unknown-model",
        pooling_strategy=PoolingStrategy.CLS,
        normalize_embeddings=True,
    )

    assert config.pooling_strategy is PoolingStrategy.CLS
    assert config.normalize_embeddings is True


def test_bge_small_auto_uses_profile_embedding_config() -> None:
    config = resolve_embedding_config("BAAI/bge-small-en-v1.5")

    assert config.pooling_strategy is PoolingStrategy.CLS
    assert config.normalize_embeddings is True


def test_bge_small_allows_matching_explicit_embedding_config() -> None:
    config = resolve_embedding_config(
        "BAAI/bge-small-en-v1.5",
        pooling_strategy="cls",
        normalize_embeddings=True,
    )

    assert config.pooling_strategy is PoolingStrategy.CLS
    assert config.normalize_embeddings is True


def test_bge_small_rejects_incorrect_explicit_pooling() -> None:
    with pytest.raises(ValueError, match="pooling_strategy must be 'cls', got 'mean'"):
        resolve_embedding_config(
            "BAAI/bge-small-en-v1.5",
            pooling_strategy=PoolingStrategy.MEAN,
            normalize_embeddings=AutoSetting.AUTO,
        )


def test_bge_small_rejects_incorrect_explicit_normalization() -> None:
    with pytest.raises(ValueError, match="normalize_embeddings must be True, got False"):
        resolve_embedding_config(
            "BAAI/bge-small-en-v1.5",
            pooling_strategy=PoolingStrategy.AUTO,
            normalize_embeddings=False,
        )


def test_cache_namespace_uses_resolved_embedding_values() -> None:
    config = resolve_embedding_config("BAAI/bge-small-en-v1.5")

    assert (
        config.cache_namespace(
            model_name="BAAI/bge-small-en-v1.5",
            model_revision="a" * 40,
            implementation_name="huggingface_embedding",
            implementation_version="1.0.0",
            max_length=512,
        )
        == "model=BAAI/bge-small-en-v1.5|"
        "revision=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa|"
        "implementation=huggingface_embedding@1.0.0|"
        "pooling=cls|normalize=True|max_length=512"
    )
