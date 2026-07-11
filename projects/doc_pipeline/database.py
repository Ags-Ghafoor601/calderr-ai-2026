"""
Database Module — SQLite storage for processed documents
==========================================================
Uses aiosqlite for async database operations.
Stores document metadata and extraction results as JSON.
"""

import json
import aiosqlite
from pathlib import Path
from typing import Optional

DB_PATH = Path(__file__).parent / "documents.db"


async def init_db(db_path: str | Path = DB_PATH):
    """Initialize the database and create tables if they don't exist."""
    async with aiosqlite.connect(str(db_path)) as db:
        await db.execute("""
            CREATE TABLE IF NOT EXISTS documents (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                filename TEXT NOT NULL,
                file_type TEXT NOT NULL,
                file_size_bytes INTEGER NOT NULL,
                uploaded_at TEXT NOT NULL,
                summary TEXT,
                key_terms TEXT,
                entities TEXT,
                dates TEXT,
                action_items TEXT,
                document_type_guess TEXT,
                language TEXT,
                word_count INTEGER,
                processing_time_ms REAL,
                status TEXT DEFAULT 'completed',
                error TEXT
            )
        """)
        await db.commit()


async def save_document(record: dict, db_path: str | Path = DB_PATH) -> int:
    """
    Save a processed document to the database.

    Args:
        record: Dictionary with document data (from DocumentRecord.model_dump())

    Returns:
        The database ID of the inserted record
    """
    extraction = record.get("extraction", {})

    async with aiosqlite.connect(str(db_path)) as db:
        cursor = await db.execute(
            """
            INSERT INTO documents (
                filename, file_type, file_size_bytes, uploaded_at,
                summary, key_terms, entities, dates, action_items,
                document_type_guess, language, word_count,
                processing_time_ms, status, error
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.get("filename", "unknown"),
                record.get("file_type", "txt"),
                record.get("file_size_bytes", 0),
                record.get("uploaded_at", ""),
                extraction.get("summary", ""),
                json.dumps(extraction.get("key_terms", [])),
                json.dumps(extraction.get("entities", []), default=str),
                json.dumps(extraction.get("dates", []), default=str),
                json.dumps(extraction.get("action_items", []), default=str),
                extraction.get("document_type_guess", "general"),
                extraction.get("language", "English"),
                extraction.get("word_count"),
                record.get("processing_time_ms", 0),
                record.get("status", "completed"),
                record.get("error"),
            ),
        )
        await db.commit()
        return cursor.lastrowid


async def get_document(doc_id: int, db_path: str | Path = DB_PATH) -> Optional[dict]:
    """Fetch a single document by ID."""
    async with aiosqlite.connect(str(db_path)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM documents WHERE id = ?", (doc_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return _row_to_dict(row)


async def get_all_documents(
    db_path: str | Path = DB_PATH,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Fetch all documents with pagination."""
    async with aiosqlite.connect(str(db_path)) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM documents ORDER BY id DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(row) for row in rows]


async def get_document_count(db_path: str | Path = DB_PATH) -> int:
    """Get total number of documents in the database."""
    async with aiosqlite.connect(str(db_path)) as db:
        cursor = await db.execute("SELECT COUNT(*) FROM documents")
        row = await cursor.fetchone()
        return row[0] if row else 0


async def delete_document(doc_id: int, db_path: str | Path = DB_PATH) -> bool:
    """Delete a document by ID. Returns True if deleted."""
    async with aiosqlite.connect(str(db_path)) as db:
        cursor = await db.execute(
            "DELETE FROM documents WHERE id = ?", (doc_id,)
        )
        await db.commit()
        return cursor.rowcount > 0


def _row_to_dict(row) -> dict:
    """Convert a database row to a dictionary with parsed JSON fields."""
    d = dict(row)
    # Parse JSON fields back into Python objects
    for field in ["key_terms", "entities", "dates", "action_items"]:
        if field in d and d[field]:
            try:
                d[field] = json.loads(d[field])
            except (json.JSONDecodeError, TypeError):
                d[field] = []
    return d
