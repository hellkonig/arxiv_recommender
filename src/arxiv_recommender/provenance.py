"""Shared recommendation provenance contracts."""

from enum import Enum

from pydantic import BaseModel, Field, JsonValue, field_validator


class SelectionSource(str, Enum):
    """Reason a recommendation was selected for display."""

    BASE_RANKER = "base_ranker"
    PERSONAL_RANKER = "personal_ranker"
    EXPLORATION = "exploration"


class ModelProvenance(BaseModel):
    """Reproducible identity and configuration for a recommendation model."""

    name: str = Field(min_length=1)
    version: str = Field(min_length=1)
    config: dict[str, JsonValue] = Field(default_factory=dict)

    model_config = {"frozen": True}

    @field_validator("name", "version")
    @classmethod
    def validate_non_empty_identity(cls, value: str) -> str:
        """Normalize and reject blank model identity fields."""
        normalized_value = value.strip()
        if not normalized_value:
            raise ValueError("Model name and version must not be blank.")
        return normalized_value
