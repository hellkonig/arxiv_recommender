from typing import Any

from pydantic import BaseModel, Field

from arxiv_recommender.provenance import ModelProvenance, SelectionSource
from arxiv_recommender.schemas import Paper


class RecommendationItem(BaseModel):
    """A ranked paper recommendation."""

    paper: Paper
    score: float
    selection_source: SelectionSource

    model_config = {"frozen": True}


class RecommendationRunResult(BaseModel):
    """Structured output for a recommendation pipeline execution."""

    recommendations: list[RecommendationItem] = Field(default_factory=list)
    metrics_summary: dict[str, Any] = Field(default_factory=dict)
    favorite_papers_count: int
    candidate_papers_count: int
    embedding_provenance: ModelProvenance
    ranker_provenance: ModelProvenance
