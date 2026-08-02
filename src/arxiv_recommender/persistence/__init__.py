"""SQLite persistence helpers."""

from arxiv_recommender.persistence.database import connect_database
from arxiv_recommender.persistence.migrations import MigrationError, apply_migrations
from arxiv_recommender.persistence.models import (
    DisplayedRecommendation,
    ModelKind,
    ModelVersionSpec,
    RecommendationRunRecord,
    SelectionSource,
    StoredImpression,
    StoredRecommendationRun,
)
from arxiv_recommender.persistence.repository import (
    RecommendationPersistenceError,
    RecommendationRepository,
)

__all__ = [
    "DisplayedRecommendation",
    "MigrationError",
    "ModelKind",
    "ModelVersionSpec",
    "RecommendationPersistenceError",
    "RecommendationRepository",
    "RecommendationRunRecord",
    "SelectionSource",
    "StoredImpression",
    "StoredRecommendationRun",
    "apply_migrations",
    "connect_database",
]
