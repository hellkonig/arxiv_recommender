from pydantic import BaseModel, Field, field_validator

from arxiv_recommender.text_vectorization.config import (
    AutoSetting,
    FULL_COMMIT_SHA_PATTERN,
    NormalizeEmbeddingsSetting,
    PoolingStrategy,
)
from arxiv_recommender.utils.logging import SUPPORTED_LOG_LEVELS


class VectorizerConfig(BaseModel):
    module_name: str = Field(description="Module name for the vectorizer")
    class_name: str = Field(description="Class name for the vectorizer")
    model_name: str = Field(description="Path or name of the model")
    model_revision: str = Field(
        pattern=FULL_COMMIT_SHA_PATTERN,
        description="Full 40-character lowercase model artifact commit SHA",
    )
    cache_size: int = Field(
        default=1000,
        ge=0,
        description="Maximum number of embeddings to cache",
    )
    pooling_strategy: PoolingStrategy = Field(
        default=PoolingStrategy.AUTO,
        description="Embedding pooling strategy, or auto for known model profiles",
    )
    normalize_embeddings: NormalizeEmbeddingsSetting = Field(
        default=AutoSetting.AUTO,
        description="Whether to L2-normalize embeddings, or auto for known model profiles",
    )
    max_length: int = Field(
        default=512,
        gt=0,
        description="Maximum token length for embedding model inputs",
    )

    model_config = {"frozen": True}


class AppConfig(BaseModel):
    favorite_papers_path: str = Field(description="Path to the favorite papers JSON file")
    vectorizer: VectorizerConfig = Field(description="Vectorizer configuration")
    top_k: int = Field(default=10, gt=0, description="Number of top recommendations to return")
    log_level: str = Field(
        default="INFO",
        description=f"Logging level ({', '.join(SUPPORTED_LOG_LEVELS)})",
    )

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized_value = value.upper()
        if normalized_value not in SUPPORTED_LOG_LEVELS:
            raise ValueError(f"log_level must be one of: {', '.join(SUPPORTED_LOG_LEVELS)}")
        return normalized_value

    model_config = {"frozen": True}
