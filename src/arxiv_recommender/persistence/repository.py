from datetime import datetime, timezone
import json
import sqlite3

from arxiv_recommender.persistence.models import (
    DisplayedRecommendation,
    ModelVersionSpec,
    RecommendationRunRecord,
    StoredImpression,
    StoredRecommendationRun,
)
from arxiv_recommender.schemas import Paper


class RecommendationPersistenceError(RuntimeError):
    """Raised when a recommendation run cannot be persisted atomically."""


class RecommendationRepository:
    """Persist recommendation runs and displayed papers in SQLite."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        """Initialize the repository with an open, migrated connection.

        Args:
            connection: SQLite connection whose schema migrations have already
                been applied.
        """
        self._connection = connection

    def record_recommendation_run(
        self,
        record: RecommendationRunRecord,
    ) -> StoredRecommendationRun:
        """Persist a recommendation run and its displayed results atomically.

        Args:
            record: Validated run metadata, model provenance, and displayed
                recommendations.

        Returns:
            Database identifiers for the stored run and impressions.

        Raises:
            RecommendationPersistenceError: If the connection already has an
                active transaction or any database write fails.
        """
        if self._connection.in_transaction:
            raise RecommendationPersistenceError(
                "Cannot record a recommendation run during an active transaction."
            )

        try:
            embedding_config_json = _canonical_json(record.embedding_model.config)
            ranker_config_json = _canonical_json(record.ranker_model.config)
            metrics_json = _canonical_json(record.metrics)
        except (TypeError, ValueError) as exc:
            raise RecommendationPersistenceError(
                "Recommendation run contains data that cannot be serialized as JSON."
            ) from exc

        event_timestamp = record.run_completed_at or record.run_started_at

        try:
            self._connection.execute("BEGIN")
            embedding_model_id = self._upsert_model_version(
                record.embedding_model,
                embedding_config_json,
                event_timestamp,
            )
            ranker_model_id = self._upsert_model_version(
                record.ranker_model,
                ranker_config_json,
                event_timestamp,
            )
            run_id = self._insert_recommendation_run(
                record,
                embedding_model_id,
                ranker_model_id,
                metrics_json,
            )
            stored_impressions = [
                self._insert_displayed_recommendation(
                    run_id,
                    recommendation,
                    event_timestamp,
                )
                for recommendation in record.displayed_recommendations
            ]
            self._connection.execute("COMMIT")
        except RecommendationPersistenceError:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise
        except (sqlite3.Error, ValueError) as exc:
            if self._connection.in_transaction:
                self._connection.execute("ROLLBACK")
            raise RecommendationPersistenceError("Failed to persist recommendation run.") from exc

        return StoredRecommendationRun(run_id=run_id, impressions=stored_impressions)

    def _upsert_model_version(
        self,
        model: ModelVersionSpec,
        config_json: str,
        created_at: datetime,
    ) -> int:
        self._connection.execute(
            """
            INSERT INTO model_versions(model_kind, name, version, config_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(model_kind, name, version, config_json) DO NOTHING
            """,
            (
                model.model_kind.value,
                model.name,
                model.version,
                config_json,
                _utc_isoformat(created_at),
            ),
        )
        row = self._connection.execute(
            """
            SELECT id
            FROM model_versions
            WHERE model_kind = ? AND name = ? AND version = ? AND config_json = ?
            """,
            (model.model_kind.value, model.name, model.version, config_json),
        ).fetchone()
        if row is None:
            raise RecommendationPersistenceError("Stored model version could not be loaded.")
        return int(row[0])

    def _insert_recommendation_run(
        self,
        record: RecommendationRunRecord,
        embedding_model_id: int,
        ranker_model_id: int,
        metrics_json: str,
    ) -> int:
        cursor = self._connection.execute(
            """
            INSERT INTO recommendation_runs(
                run_started_at,
                run_completed_at,
                requested_date,
                favorite_papers_count,
                candidate_papers_count,
                top_k,
                embedding_model_version_id,
                ranker_model_version_id,
                metrics_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                _utc_isoformat(record.run_started_at),
                (
                    _utc_isoformat(record.run_completed_at)
                    if record.run_completed_at is not None
                    else None
                ),
                record.requested_date,
                record.favorite_papers_count,
                record.candidate_papers_count,
                record.top_k,
                embedding_model_id,
                ranker_model_id,
                metrics_json,
            ),
        )
        return _last_inserted_id(cursor, "recommendation run")

    def _insert_displayed_recommendation(
        self,
        run_id: int,
        recommendation: DisplayedRecommendation,
        created_at: datetime,
    ) -> StoredImpression:
        paper_id = self._upsert_paper(recommendation.paper, created_at)
        cursor = self._connection.execute(
            """
            INSERT INTO impressions(
                recommendation_run_id,
                paper_id,
                displayed_rank,
                score,
                selection_source,
                created_at
            )
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                paper_id,
                recommendation.displayed_rank,
                recommendation.score,
                recommendation.selection_source.value,
                _utc_isoformat(created_at),
            ),
        )
        return StoredImpression(
            impression_id=_last_inserted_id(cursor, "impression"),
            paper_id=paper_id,
            displayed_rank=recommendation.displayed_rank,
        )

    def _upsert_paper(self, paper: Paper, seen_at: datetime) -> int:
        arxiv_id = paper.arxiv_id
        if arxiv_id is None:
            raise RecommendationPersistenceError(
                "Displayed recommendations require a non-empty arxiv_id."
            )

        seen_at_text = _utc_isoformat(seen_at)
        self._connection.execute(
            """
            INSERT INTO papers(
                arxiv_id,
                url,
                title,
                abstract,
                authors_json,
                categories_json,
                published_at,
                updated_at,
                created_at,
                last_seen_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(arxiv_id) DO UPDATE SET
                url = excluded.url,
                title = excluded.title,
                abstract = excluded.abstract,
                authors_json = excluded.authors_json,
                categories_json = excluded.categories_json,
                published_at = excluded.published_at,
                updated_at = excluded.updated_at,
                last_seen_at = excluded.last_seen_at
            """,
            (
                arxiv_id,
                paper.url,
                paper.title,
                paper.abstract,
                _canonical_json(paper.authors),
                _canonical_json(paper.categories),
                _optional_utc_isoformat(paper.published),
                _optional_utc_isoformat(paper.updated),
                seen_at_text,
                seen_at_text,
            ),
        )
        row = self._connection.execute(
            "SELECT id FROM papers WHERE arxiv_id = ?",
            (arxiv_id,),
        ).fetchone()
        if row is None:
            raise RecommendationPersistenceError("Stored paper could not be loaded.")
        return int(row[0])


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _utc_isoformat(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _optional_utc_isoformat(value: datetime | None) -> str | None:
    return _utc_isoformat(value) if value is not None else None


def _last_inserted_id(cursor: sqlite3.Cursor, record_name: str) -> int:
    row_id = cursor.lastrowid
    if row_id is None:
        raise RecommendationPersistenceError(f"Could not determine stored {record_name} ID.")
    return int(row_id)
