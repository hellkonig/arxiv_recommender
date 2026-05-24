from typing import Any

from pydantic import BaseModel, Field


class RecommendationRunResult(BaseModel):
    """Structured output for a recommendation pipeline execution."""

    recommendations: list[dict[str, Any]] = Field(default_factory=list)
    metrics_summary: dict[str, Any] = Field(default_factory=dict)
    favorite_papers_count: int
    candidate_papers_count: int
