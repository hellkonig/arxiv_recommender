from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field, JsonValue, field_validator, model_validator

from arxiv_recommender.schemas import Paper


class ModelKind(str, Enum):
    """Supported persisted model categories."""

    EMBEDDING = "embedding"
    RANKER = "ranker"


class SelectionSource(str, Enum):
    """Reason a recommendation was selected for display."""

    BASE_RANKER = "base_ranker"
    PERSONAL_RANKER = "personal_ranker"
    EXPLORATION = "exploration"


class ModelVersionSpec(BaseModel):
    """Validated identity and configuration for a persisted model version."""

    model_kind: ModelKind
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


class DisplayedRecommendation(BaseModel):
    """A recommendation that was selected for display."""

    paper: Paper
    displayed_rank: int = Field(gt=0)
    score: float = Field(allow_inf_nan=False)
    selection_source: SelectionSource

    model_config = {"frozen": True}

    @model_validator(mode="after")
    def validate_paper_identity(self) -> "DisplayedRecommendation":
        """Require a stable paper identity before persistence."""
        if self.paper.arxiv_id is None or not self.paper.arxiv_id.strip():
            raise ValueError("Displayed recommendations require a non-empty arxiv_id.")
        for field_name, timestamp in (
            ("published", self.paper.published),
            ("updated", self.paper.updated),
        ):
            if timestamp is not None and timestamp.utcoffset() is None:
                raise ValueError(f"Paper {field_name} timestamp must be timezone-aware.")
        return self


class RecommendationRunRecord(BaseModel):
    """Validated recommendation run ready for atomic persistence."""

    run_started_at: datetime
    run_completed_at: datetime | None = None
    requested_date: str | None = Field(default=None, pattern=r"^\d{8}$")
    favorite_papers_count: int = Field(ge=0)
    candidate_papers_count: int = Field(ge=0)
    top_k: int = Field(gt=0)
    embedding_model: ModelVersionSpec
    ranker_model: ModelVersionSpec
    metrics: dict[str, JsonValue] = Field(default_factory=dict)
    displayed_recommendations: list[DisplayedRecommendation] = Field(default_factory=list)

    model_config = {"frozen": True}

    @field_validator("run_started_at", "run_completed_at")
    @classmethod
    def validate_timezone_aware_timestamp(cls, value: datetime | None) -> datetime | None:
        """Require unambiguous timestamps for chronological evaluation."""
        if value is not None and value.utcoffset() is None:
            raise ValueError("Recommendation run timestamps must be timezone-aware.")
        return value

    @field_validator("requested_date")
    @classmethod
    def validate_requested_date(cls, value: str | None) -> str | None:
        """Require a real calendar date when a requested date is present."""
        if value is not None:
            try:
                datetime.strptime(value, "%Y%m%d")
            except ValueError as exc:
                raise ValueError("requested_date must be a valid date in YYYYMMDD format.") from exc
        return value

    @model_validator(mode="after")
    def validate_run_contract(self) -> "RecommendationRunRecord":
        """Validate cross-field constraints for a persistable run."""
        if self.embedding_model.model_kind is not ModelKind.EMBEDDING:
            raise ValueError("embedding_model must have model_kind='embedding'.")
        if self.ranker_model.model_kind is not ModelKind.RANKER:
            raise ValueError("ranker_model must have model_kind='ranker'.")
        if self.run_completed_at is not None and self.run_completed_at < self.run_started_at:
            raise ValueError("run_completed_at must not be earlier than run_started_at.")

        recommendation_count = len(self.displayed_recommendations)
        if recommendation_count > self.top_k:
            raise ValueError("Displayed recommendation count must not exceed top_k.")
        if recommendation_count > self.candidate_papers_count:
            raise ValueError(
                "Displayed recommendation count must not exceed candidate_papers_count."
            )

        displayed_ranks = [
            recommendation.displayed_rank for recommendation in self.displayed_recommendations
        ]
        expected_ranks = list(range(1, recommendation_count + 1))
        if displayed_ranks != expected_ranks:
            raise ValueError("Displayed recommendations must have sequential ranks starting at 1.")

        arxiv_ids = [
            recommendation.paper.arxiv_id for recommendation in self.displayed_recommendations
        ]
        if len(arxiv_ids) != len(set(arxiv_ids)):
            raise ValueError("A paper cannot be displayed more than once in the same run.")

        return self


class StoredImpression(BaseModel):
    """Database identifiers assigned to one stored impression."""

    impression_id: int = Field(gt=0)
    paper_id: int = Field(gt=0)
    displayed_rank: int = Field(gt=0)

    model_config = {"frozen": True}


class StoredRecommendationRun(BaseModel):
    """Database identifiers assigned to a stored recommendation run."""

    run_id: int = Field(gt=0)
    impressions: list[StoredImpression] = Field(default_factory=list)

    model_config = {"frozen": True}
