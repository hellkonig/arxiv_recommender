import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from arxiv_recommender.persistence import MigrationError, apply_migrations, connect_database
from arxiv_recommender.persistence.migrations import Migration


class TestMigrations(unittest.TestCase):
    def _connect_temp_database(self) -> sqlite3.Connection:
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        database_path = Path(temp_dir.name) / "recommendations.sqlite"
        connection = connect_database(database_path)
        self.addCleanup(connection.close)
        return connection

    def test_apply_migrations_creates_initial_schema(self) -> None:
        connection = self._connect_temp_database()

        apply_migrations(connection)

        table_names = {
            row["name"]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
        self.assertEqual(
            {
                "schema_migrations",
                "papers",
                "model_versions",
                "recommendation_runs",
                "impressions",
                "feedback",
            },
            table_names,
        )

    def test_apply_migrations_is_idempotent(self) -> None:
        connection = self._connect_temp_database()

        apply_migrations(connection)
        apply_migrations(connection)

        migration_count = connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
        self.assertEqual(1, migration_count)

    def test_apply_migrations_rejects_checksum_mismatch(self) -> None:
        connection = self._connect_temp_database()
        original_migration = Migration(
            version=1,
            name="test",
            sql="CREATE TABLE example (id INTEGER PRIMARY KEY);",
        )
        changed_migration = Migration(
            version=1,
            name="test",
            sql="CREATE TABLE example (id INTEGER PRIMARY KEY, name TEXT);",
        )

        apply_migrations(connection, migrations=[original_migration])

        with self.assertRaisesRegex(MigrationError, "checksum does not match"):
            apply_migrations(connection, migrations=[changed_migration])

    def test_connection_enforces_foreign_keys(self) -> None:
        connection = self._connect_temp_database()

        apply_migrations(connection)

        with self.assertRaises(sqlite3.IntegrityError):
            connection.execute(
                """
                INSERT INTO feedback(impression_id, value, created_at)
                VALUES (?, ?, ?)
                """,
                (999, "interested", "2026-07-20T00:00:00+00:00"),
            )

    def test_initial_schema_supports_feedback_event_flow(self) -> None:
        connection = self._connect_temp_database()
        timestamp = "2026-07-20T00:00:00+00:00"

        apply_migrations(connection)

        paper_id = connection.execute(
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
            """,
            (
                "2607.12345",
                "https://arxiv.org/abs/2607.12345",
                "A Test Paper",
                "A test abstract.",
                json.dumps(["Ada Lovelace"]),
                json.dumps(["cs.IR"]),
                timestamp,
                timestamp,
                timestamp,
                timestamp,
            ),
        ).lastrowid
        embedding_model_id = connection.execute(
            """
            INSERT INTO model_versions(model_kind, name, version, config_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "embedding",
                "BAAI/bge-small-en-v1.5",
                "model=BAAI/bge-small-en-v1.5|pooling=cls|normalize=True|max_length=512",
                json.dumps({"pooling_strategy": "cls", "normalize_embeddings": True}),
                timestamp,
            ),
        ).lastrowid
        ranker_model_id = connection.execute(
            """
            INSERT INTO model_versions(model_kind, name, version, config_json, created_at)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                "ranker",
                "base_similarity",
                "1",
                json.dumps({"score_policy": "max_cosine_similarity_to_favorites"}),
                timestamp,
            ),
        ).lastrowid
        recommendation_run_id = connection.execute(
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
                timestamp,
                timestamp,
                "20260720",
                2,
                25,
                10,
                embedding_model_id,
                ranker_model_id,
                json.dumps({"embedding_latency": []}),
            ),
        ).lastrowid
        impression_id = connection.execute(
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
            (recommendation_run_id, paper_id, 1, 0.87, "base_ranker", timestamp),
        ).lastrowid

        connection.execute(
            """
            INSERT INTO feedback(impression_id, value, created_at)
            VALUES (?, ?, ?)
            """,
            (impression_id, "interested", timestamp),
        )
        connection.execute(
            """
            INSERT INTO feedback(impression_id, value, created_at)
            VALUES (?, ?, ?)
            """,
            (impression_id, "not_interested", "2026-07-20T00:05:00+00:00"),
        )

        feedback_values = [
            row["value"]
            for row in connection.execute(
                "SELECT value FROM feedback WHERE impression_id = ? ORDER BY created_at",
                (impression_id,),
            ).fetchall()
        ]
        self.assertEqual(["interested", "not_interested"], feedback_values)


if __name__ == "__main__":
    unittest.main()
