from dataclasses import dataclass
from enum import Enum


class PoolingStrategy(str, Enum):
    """Supported token pooling strategies for HuggingFace hidden states."""

    AUTO = "auto"
    MEAN = "mean"
    CLS = "cls"


class AutoSetting(str, Enum):
    """Automatic configuration sentinel for non-string embedding settings."""

    AUTO = "auto"


NormalizeEmbeddings = bool | AutoSetting


@dataclass(frozen=True)
class EmbeddingProfile:
    """Embedding contract for a known model."""

    pooling_strategy: PoolingStrategy
    normalize_embeddings: bool


@dataclass(frozen=True)
class ResolvedEmbeddingConfig:
    """Concrete embedding settings after resolving auto values."""

    pooling_strategy: PoolingStrategy
    normalize_embeddings: bool

    def cache_namespace(self, model_name: str, max_length: int) -> str:
        """Return a stable cache namespace for this embedding configuration."""
        return (
            f"model={model_name}|pooling={self.pooling_strategy.value}|"
            f"normalize={self.normalize_embeddings}|max_length={max_length}"
        )


MODEL_EMBEDDING_PROFILES = {
    "BAAI/bge-small-en-v1.5": EmbeddingProfile(
        pooling_strategy=PoolingStrategy.CLS,
        normalize_embeddings=True,
    ),
}


def resolve_embedding_config(
    model_name: str,
    pooling_strategy: PoolingStrategy | str = PoolingStrategy.AUTO,
    normalize_embeddings: NormalizeEmbeddings | str = AutoSetting.AUTO,
) -> ResolvedEmbeddingConfig:
    """Resolve and validate model-specific embedding settings.

    Args:
        model_name: HuggingFace model path or identifier.
        pooling_strategy: Requested pooling strategy, or auto.
        normalize_embeddings: Requested L2-normalization setting, or auto.

    Returns:
        Concrete embedding settings.

    Raises:
        ValueError: If a known model is configured against its embedding contract.
    """
    requested_pooling = PoolingStrategy(pooling_strategy)
    requested_normalize = _coerce_normalize_embeddings(normalize_embeddings)

    profile = MODEL_EMBEDDING_PROFILES.get(model_name)
    default_pooling = profile.pooling_strategy if profile else PoolingStrategy.MEAN
    default_normalize = profile.normalize_embeddings if profile else False

    resolved_config = ResolvedEmbeddingConfig(
        pooling_strategy=default_pooling
        if requested_pooling is PoolingStrategy.AUTO
        else requested_pooling,
        normalize_embeddings=default_normalize
        if requested_normalize is AutoSetting.AUTO
        else bool(requested_normalize),
    )

    if profile:
        _validate_known_model_config(model_name, profile, resolved_config)

    return resolved_config


def _coerce_normalize_embeddings(value: NormalizeEmbeddings | str) -> NormalizeEmbeddings:
    if isinstance(value, bool):
        return value
    return AutoSetting(value)


def _validate_known_model_config(
    model_name: str,
    profile: EmbeddingProfile,
    resolved_config: ResolvedEmbeddingConfig,
) -> None:
    errors = []
    if resolved_config.pooling_strategy is not profile.pooling_strategy:
        errors.append(
            "pooling_strategy must be "
            f"'{profile.pooling_strategy.value}', got '{resolved_config.pooling_strategy.value}'"
        )
    if resolved_config.normalize_embeddings != profile.normalize_embeddings:
        errors.append(
            "normalize_embeddings must be "
            f"{profile.normalize_embeddings}, got {resolved_config.normalize_embeddings}"
        )

    if errors:
        raise ValueError(f"{model_name} embedding config is incompatible: {'; '.join(errors)}.")
