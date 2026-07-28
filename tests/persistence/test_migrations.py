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
        connection: sqlite3.Connection = connect_database(database_path)
        self.addCleanup(connection.close)
        return connection

    def test_apply_migrations_loads_packaged_sql_files_by_default(self) -> None:
        """Default migration application uses the real packaged SQL files."""
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

    def test_apply_migrations_is_idempotent_for_already_applied_migrations(self) -> None:
        """Running migrations twice should not rerun SQL already recorded."""
        connection = self._connect_temp_database()

        apply_migrations(connection)
        first_applied_row = connection.execute(
            "SELECT version, applied_at FROM schema_migrations"
        ).fetchone()
        apply_migrations(connection)

        second_applied_row = connection.execute(
            "SELECT version, applied_at FROM schema_migrations"
        ).fetchone()
        migration_count = connection.execute("SELECT COUNT(*) FROM schema_migrations").fetchone()[0]
        self.assertEqual(1, migration_count)
        self.assertEqual(first_applied_row["version"], second_applied_row["version"])
        self.assertEqual(first_applied_row["applied_at"], second_applied_row["applied_at"])

    def test_apply_migrations_runs_pending_migrations_in_order(self) -> None:
        connection = self._connect_temp_database()
        initial_migration = Migration(
            version=1,
            name="create_example",
            sql="CREATE TABLE example (id INTEGER PRIMARY KEY);",
        )
        pending_migration = Migration(
            version=2,
            name="add_example_name",
            sql="ALTER TABLE example ADD COLUMN name TEXT;",
        )

        apply_migrations(connection, migrations=[initial_migration])
        apply_migrations(connection, migrations=[initial_migration, pending_migration])

        migration_versions = [
            row["version"]
            for row in connection.execute(
                "SELECT version FROM schema_migrations ORDER BY version"
            ).fetchall()
        ]
        example_columns = [
            row["name"] for row in connection.execute("PRAGMA table_info(example)").fetchall()
        ]
        self.assertEqual([1, 2], migration_versions)
        self.assertEqual(["id", "name"], example_columns)

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

    def test_schema_rejects_feedback_with_unknown_impression_id(self) -> None:
        """Feedback with a nonexistent impression_id must be rejected."""
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

    def test_initial_schema_supports_recommendation_feedback_event_flow(self) -> None:
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

        stored_paper = connection.execute(
            "SELECT title FROM papers WHERE id = ?", (paper_id,)
        ).fetchone()
        stored_run = connection.execute(
            """
            SELECT embedding_model_version_id, ranker_model_version_id
            FROM recommendation_runs
            WHERE id = ?
            """,
            (recommendation_run_id,),
        ).fetchone()
        stored_impression = connection.execute(
            """
            SELECT recommendation_run_id, paper_id, displayed_rank, selection_source
            FROM impressions
            WHERE id = ?
            """,
            (impression_id,),
        ).fetchone()
        feedback_values = [
            row["value"]
            for row in connection.execute(
                "SELECT value FROM feedback WHERE impression_id = ? ORDER BY created_at",
                (impression_id,),
            ).fetchall()
        ]
        self.assertEqual("A Test Paper", stored_paper["title"])
        self.assertEqual(embedding_model_id, stored_run["embedding_model_version_id"])
        self.assertEqual(ranker_model_id, stored_run["ranker_model_version_id"])
        self.assertEqual(recommendation_run_id, stored_impression["recommendation_run_id"])
        self.assertEqual(paper_id, stored_impression["paper_id"])
        self.assertEqual(1, stored_impression["displayed_rank"])
        self.assertEqual("base_ranker", stored_impression["selection_source"])
        self.assertEqual(["interested", "not_interested"], feedback_values)


if __name__ == "__main__":
    unittest.main()
