from datetime import datetime, timedelta, timezone
import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from pydantic import JsonValue, ValidationError

from arxiv_recommender.persistence import (
    DisplayedRecommendation,
    ModelKind,
    ModelVersionSpec,
    RecommendationPersistenceError,
    RecommendationRepository,
    RecommendationRunRecord,
    SelectionSource,
    apply_migrations,
    connect_database,
)
from arxiv_recommender.schemas import Paper


class TestRecommendationRepository(unittest.TestCase):
    def setUp(self) -> None:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        database_path = Path(temp_dir.name) / "recommendations.sqlite"
        self.connection: sqlite3.Connection = connect_database(database_path)
        self.addCleanup(self.connection.close)
        apply_migrations(self.connection)
        self.repository = RecommendationRepository(self.connection)
        self.started_at = datetime(2026, 7, 29, 9, 0, tzinfo=timezone.utc)
        self.completed_at = self.started_at + timedelta(seconds=8)

    def _embedding_model(
        self,
        config: dict[str, JsonValue] | None = None,
    ) -> ModelVersionSpec:
        return ModelVersionSpec(
            model_kind=ModelKind.EMBEDDING,
            name="BAAI/bge-small-en-v1.5",
            version="bge-small-cls-normalized-v1",
            config=config
            or {
                "pooling_strategy": "cls",
                "normalize_embeddings": True,
                "max_length": 512,
            },
        )

    def _ranker_model(
        self,
        config: dict[str, JsonValue] | None = None,
    ) -> ModelVersionSpec:
        return ModelVersionSpec(
            model_kind=ModelKind.RANKER,
            name="max_favorite_cosine_similarity",
            version="1",
            config=config
            or {
                "similarity": "cosine",
                "favorite_aggregation": "max",
            },
        )

    def _displayed_recommendation(
        self,
        rank: int,
        arxiv_id: str | None = None,
        title: str | None = None,
    ) -> DisplayedRecommendation:
        paper_id = arxiv_id or f"2607.{rank:05d}"
        return DisplayedRecommendation(
            paper=Paper(
                arxiv_id=paper_id,
                url=f"https://arxiv.org/abs/{paper_id}",
                title=title or f"Paper {rank}",
                abstract=f"Abstract {rank}",
                authors=[f"Author {rank}"],
                categories=["cs.IR"],
                published=self.started_at - timedelta(days=1),
                updated=self.started_at,
            ),
            displayed_rank=rank,
            score=0.9 - (rank * 0.1),
            selection_source=SelectionSource.BASE_RANKER,
        )

    def _run_record(
        self,
        recommendations: list[DisplayedRecommendation] | None = None,
        embedding_model: ModelVersionSpec | None = None,
        ranker_model: ModelVersionSpec | None = None,
    ) -> RecommendationRunRecord:
        displayed_recommendations = recommendations
        if displayed_recommendations is None:
            displayed_recommendations = [
                self._displayed_recommendation(rank) for rank in range(1, 4)
            ]
        return RecommendationRunRecord(
            run_started_at=self.started_at,
            run_completed_at=self.completed_at,
            requested_date="20260728",
            favorite_papers_count=2,
            candidate_papers_count=25,
            top_k=3,
            embedding_model=embedding_model or self._embedding_model(),
            ranker_model=ranker_model or self._ranker_model(),
            metrics={"cache": {"hits": 2, "misses": 25}},
            displayed_recommendations=displayed_recommendations,
        )

    def _table_count(self, table_name: str) -> int:
        allowed_tables = {
            "papers",
            "model_versions",
            "recommendation_runs",
            "impressions",
        }
        if table_name not in allowed_tables:
            raise ValueError(f"Unsupported test table: {table_name}")
        row = self.connection.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()
        return int(row[0])

    def test_records_run_and_only_displayed_papers(self) -> None:
        record = self._run_record()

        stored_run = self.repository.record_recommendation_run(record)

        self.assertEqual(1, self._table_count("recommendation_runs"))
        self.assertEqual(2, self._table_count("model_versions"))
        self.assertEqual(3, self._table_count("papers"))
        self.assertEqual(3, self._table_count("impressions"))
        self.assertEqual(1, stored_run.run_id)
        self.assertEqual([1, 2, 3], [item.displayed_rank for item in stored_run.impressions])

        stored_database_run = self.connection.execute(
            """
            SELECT
                candidate_papers_count,
                top_k,
                embedding_model_version_id,
                ranker_model_version_id,
                metrics_json
            FROM recommendation_runs
            WHERE id = ?
            """,
            (stored_run.run_id,),
        ).fetchone()
        self.assertEqual(25, stored_database_run["candidate_papers_count"])
        self.assertEqual(3, stored_database_run["top_k"])
        self.assertNotEqual(
            stored_database_run["embedding_model_version_id"],
            stored_database_run["ranker_model_version_id"],
        )
        self.assertEqual(
            {"cache": {"hits": 2, "misses": 25}},
            json.loads(stored_database_run["metrics_json"]),
        )

        stored_impressions = self.connection.execute(
            """
            SELECT displayed_rank, score, selection_source
            FROM impressions
            ORDER BY displayed_rank
            """
        ).fetchall()
        self.assertEqual([1, 2, 3], [row["displayed_rank"] for row in stored_impressions])
        self.assertEqual(
            ["base_ranker", "base_ranker", "base_ranker"],
            [row["selection_source"] for row in stored_impressions],
        )
        self.assertAlmostEqual(0.8, stored_impressions[0]["score"])

    def test_reuses_paper_and_model_versions_across_runs(self) -> None:
        first_record = self._run_record(
            recommendations=[
                self._displayed_recommendation(1, arxiv_id="2607.12345", title="First title")
            ]
        )
        second_record = first_record.model_copy(
            update={
                "run_started_at": self.started_at + timedelta(days=1),
                "run_completed_at": self.completed_at + timedelta(days=1),
                "displayed_recommendations": [
                    self._displayed_recommendation(
                        1,
                        arxiv_id="2607.12345",
                        title="Updated title",
                    )
                ],
            }
        )

        first_stored_run = self.repository.record_recommendation_run(first_record)
        second_stored_run = self.repository.record_recommendation_run(second_record)

        self.assertEqual(2, self._table_count("recommendation_runs"))
        self.assertEqual(2, self._table_count("model_versions"))
        self.assertEqual(1, self._table_count("papers"))
        self.assertEqual(2, self._table_count("impressions"))
        self.assertEqual(
            first_stored_run.impressions[0].paper_id,
            second_stored_run.impressions[0].paper_id,
        )
        paper = self.connection.execute(
            "SELECT title, created_at, last_seen_at FROM papers WHERE arxiv_id = ?",
            ("2607.12345",),
        ).fetchone()
        self.assertEqual("Updated title", paper["title"])
        self.assertEqual(self.completed_at.isoformat(), paper["created_at"])
        self.assertEqual(
            (self.completed_at + timedelta(days=1)).isoformat(),
            paper["last_seen_at"],
        )

    def test_canonical_model_json_reuses_semantically_identical_config(self) -> None:
        first_embedding_model = self._embedding_model(
            {
                "pooling_strategy": "cls",
                "normalize_embeddings": True,
                "max_length": 512,
            }
        )
        second_embedding_model = self._embedding_model(
            {
                "max_length": 512,
                "normalize_embeddings": True,
                "pooling_strategy": "cls",
            }
        )

        self.repository.record_recommendation_run(
            self._run_record(embedding_model=first_embedding_model)
        )
        second_record = self._run_record(embedding_model=second_embedding_model).model_copy(
            update={
                "run_started_at": self.started_at + timedelta(days=1),
                "run_completed_at": self.completed_at + timedelta(days=1),
            }
        )
        self.repository.record_recommendation_run(second_record)

        embedding_rows = self.connection.execute(
            """
            SELECT config_json
            FROM model_versions
            WHERE model_kind = 'embedding'
            """
        ).fetchall()
        self.assertEqual(1, len(embedding_rows))
        self.assertEqual(
            ('{"max_length":512,"normalize_embeddings":true,"pooling_strategy":"cls"}'),
            embedding_rows[0]["config_json"],
        )

    def test_database_failure_rolls_back_complete_run(self) -> None:
        self.connection.execute(
            """
            CREATE TRIGGER reject_second_impression
            BEFORE INSERT ON impressions
            WHEN NEW.displayed_rank = 2
            BEGIN
                SELECT RAISE(ABORT, 'second impression rejected');
            END
            """
        )

        with self.assertRaises(RecommendationPersistenceError):
            self.repository.record_recommendation_run(self._run_record())

        self.assertEqual(0, self._table_count("recommendation_runs"))
        self.assertEqual(0, self._table_count("model_versions"))
        self.assertEqual(0, self._table_count("papers"))
        self.assertEqual(0, self._table_count("impressions"))

    def test_missing_arxiv_id_is_rejected_before_database_write(self) -> None:
        with self.assertRaisesRegex(ValidationError, "non-empty arxiv_id"):
            DisplayedRecommendation(
                paper=Paper(title="No identity", abstract="Cannot be persisted"),
                displayed_rank=1,
                score=0.5,
                selection_source=SelectionSource.BASE_RANKER,
            )

        self.assertEqual(0, self._table_count("recommendation_runs"))
        self.assertEqual(0, self._table_count("papers"))
        self.assertEqual(0, self._table_count("impressions"))

    def test_run_record_rejects_invalid_display_contract(self) -> None:
        with self.assertRaisesRegex(ValidationError, "sequential ranks"):
            self._run_record(
                recommendations=[
                    self._displayed_recommendation(1),
                    self._displayed_recommendation(3),
                ]
            )

        with self.assertRaisesRegex(ValidationError, "model_kind='embedding'"):
            self._run_record(embedding_model=self._ranker_model())

    def test_records_empty_completed_run_without_impressions(self) -> None:
        record = self._run_record(recommendations=[]).model_copy(
            update={"candidate_papers_count": 0}
        )

        stored_run = self.repository.record_recommendation_run(record)

        self.assertEqual(1, stored_run.run_id)
        self.assertEqual([], stored_run.impressions)
        self.assertEqual(1, self._table_count("recommendation_runs"))
        self.assertEqual(2, self._table_count("model_versions"))
        self.assertEqual(0, self._table_count("papers"))
        self.assertEqual(0, self._table_count("impressions"))


if __name__ == "__main__":
    unittest.main()
