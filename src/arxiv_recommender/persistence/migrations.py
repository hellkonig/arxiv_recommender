from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import resources
import re
import sqlite3


MIGRATION_PACKAGE = "arxiv_recommender.persistence.sql_migrations"


class MigrationError(RuntimeError):
    """Raised when database migrations cannot be applied safely."""


@dataclass(frozen=True)
class Migration:
    """A SQLite schema migration loaded from a SQL file."""

    version: int
    name: str
    sql: str

    @property
    def checksum(self) -> str:
        """Return the migration SQL checksum."""
        return hashlib.sha256(self.sql.encode("utf-8")).hexdigest()


def apply_migrations(
    connection: sqlite3.Connection,
    migrations: list[Migration] | None = None,
) -> None:
    """Apply all pending SQLite migrations in version order.

    The runner checks `schema_migrations` to decide which migrations have
    already run. A new database runs every packaged migration from the first
    version to the latest version. An existing database runs only migrations
    that have not been recorded yet.

    Args:
        connection: Open SQLite connection.
        migrations: Optional explicit migration list for tests. Production
            callers should omit this so packaged SQL files are used.

    Raises:
        MigrationError: If migrations are out of order, duplicated, edited
            after being applied, or fail while executing.
    """
    connection.execute("PRAGMA foreign_keys = ON")
    _ensure_migrations_table(connection)

    ordered_migrations = migrations or load_migrations()
    _validate_migration_order(ordered_migrations)

    applied = _load_applied_migrations(connection)
    for migration in ordered_migrations:
        applied_checksum = applied.get(migration.version)

        # Same version and checksum means this exact migration already ran.
        # Skip it so repeated calls only apply newly added migration files.
        if applied_checksum == migration.checksum:
            continue

        # Same version with a different checksum means a migration file was
        # edited after being applied. Refuse to continue because the database
        # history no longer matches the checked-in migration history.
        if applied_checksum is not None:
            raise MigrationError(
                f"Applied migration {migration.version} checksum does not match local file."
            )

        # No record exists for this version, so this migration is pending.
        _apply_migration(connection, migration)


def load_migrations() -> list[Migration]:
    """Load packaged SQL migration files."""
    migration_files = sorted(
        (
            path
            for path in resources.files(MIGRATION_PACKAGE).iterdir()
            if path.name.endswith(".sql")
        ),
        key=lambda path: path.name,
    )
    return [
        _migration_from_file(path.name, path.read_text(encoding="utf-8"))
        for path in migration_files
    ]


def _ensure_migrations_table(connection: sqlite3.Connection) -> None:
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS schema_migrations (
            version INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            checksum TEXT NOT NULL,
            applied_at TEXT NOT NULL
        )
        """
    )


def _load_applied_migrations(connection: sqlite3.Connection) -> dict[int, str]:
    rows = connection.execute("SELECT version, checksum FROM schema_migrations").fetchall()
    return {int(row["version"]): str(row["checksum"]) for row in rows}


def _apply_migration(connection: sqlite3.Connection, migration: Migration) -> None:
    applied_at = datetime.now(timezone.utc).isoformat()
    try:
        connection.executescript(f"BEGIN;\n{migration.sql}")
        connection.execute(
            """
            INSERT INTO schema_migrations(version, name, checksum, applied_at)
            VALUES (?, ?, ?, ?)
            """,
            (migration.version, migration.name, migration.checksum, applied_at),
        )
        connection.execute("COMMIT")
    except sqlite3.Error as exc:
        connection.execute("ROLLBACK")
        raise MigrationError(f"Failed to apply migration {migration.version}: {exc}") from exc


def _migration_from_file(filename: str, sql: str) -> Migration:
    match = re.fullmatch(r"(\d{4})_(.+)\.sql", filename)
    if not match:
        raise MigrationError(f"Invalid migration filename: {filename}")
    return Migration(version=int(match.group(1)), name=match.group(2), sql=sql)


def _validate_migration_order(migrations: list[Migration]) -> None:
    versions = [migration.version for migration in migrations]
    if versions != sorted(versions):
        raise MigrationError("Migrations must be sorted by version.")
    if len(versions) != len(set(versions)):
        raise MigrationError("Migration versions must be unique.")
