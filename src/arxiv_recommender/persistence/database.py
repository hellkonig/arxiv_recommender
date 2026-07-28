from pathlib import Path
import sqlite3


def connect_database(database_path: str | Path) -> sqlite3.Connection:
    """Open a SQLite database connection for application persistence.

    Args:
        database_path: Path to the SQLite database file.

    Returns:
        A SQLite connection configured to return rows by column name, enforce
        foreign keys, and use write-ahead logging for normal app usage.
    """
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    connection.execute("PRAGMA journal_mode = WAL")
    return connection
