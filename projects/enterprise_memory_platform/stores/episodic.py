"""
Enterprise AI Memory Platform — Episodic Memory Store
======================================================
SQLite-backed episodic memory with per-tenant isolation,
recency weighting, importance scoring, and consolidation support.
"""

import json
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import EpisodicMemory, EpisodicMemoryCreate, EpisodicQueryRequest


class EpisodicStore:
    """Per-tenant episodic memory store backed by SQLite."""

    def __init__(self, db_path: str = "data/memory_platform.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodic_memories (
                memory_id       TEXT PRIMARY KEY,
                tenant_id       TEXT NOT NULL,
                session_id      TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                content         TEXT NOT NULL,
                role            TEXT NOT NULL DEFAULT 'user',
                importance_score REAL NOT NULL DEFAULT 0.5,
                is_consolidated INTEGER NOT NULL DEFAULT 0,
                metadata        TEXT DEFAULT '{}'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ep_tenant_time
            ON episodic_memories(tenant_id, timestamp DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ep_tenant_session
            ON episodic_memories(tenant_id, session_id)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ep_importance
            ON episodic_memories(tenant_id, importance_score DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_ep_unconsolidated
            ON episodic_memories(tenant_id, is_consolidated, timestamp ASC)
        """)
        conn.commit()
        conn.close()

    def store(self, tenant_id: str, memory: EpisodicMemoryCreate) -> EpisodicMemory:
        """Store a new episodic memory for a tenant."""
        entry = EpisodicMemory(
            memory_id=str(uuid.uuid4())[:12],
            tenant_id=tenant_id,
            session_id=memory.session_id,
            content=memory.content,
            role=memory.role,
            importance_score=memory.importance_score,
            metadata=memory.metadata,
        )
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO episodic_memories
            (memory_id, tenant_id, session_id, timestamp, content, role,
             importance_score, is_consolidated, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.memory_id, entry.tenant_id, entry.session_id,
            entry.timestamp, entry.content, entry.role,
            entry.importance_score, 0, json.dumps(entry.metadata),
        ))
        conn.commit()
        conn.close()
        return entry

    def query(self, tenant_id: str, request: EpisodicQueryRequest) -> list[EpisodicMemory]:
        """Query episodic memories for a tenant."""
        conn = sqlite3.connect(self.db_path)
        sql = """
            SELECT memory_id, tenant_id, session_id, timestamp, content, role,
                   importance_score, is_consolidated, metadata
            FROM episodic_memories
            WHERE tenant_id = ?
        """
        params: list = [tenant_id]

        if request.session_id:
            sql += " AND session_id = ?"
            params.append(request.session_id)
        if request.min_importance > 0.0:
            sql += " AND importance_score >= ?"
            params.append(request.min_importance)

        sql += " ORDER BY timestamp DESC LIMIT ?"
        params.append(request.limit)

        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [self._row_to_memory(r) for r in rows]

    def get_by_id(self, tenant_id: str, memory_id: str) -> Optional[EpisodicMemory]:
        """Get a specific episodic memory."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT * FROM episodic_memories WHERE memory_id = ? AND tenant_id = ?",
            (memory_id, tenant_id),
        ).fetchone()
        conn.close()
        return self._row_to_memory(row) if row else None

    def delete(self, tenant_id: str, memory_id: str) -> bool:
        """Delete an episodic memory."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "DELETE FROM episodic_memories WHERE memory_id = ? AND tenant_id = ?",
            (memory_id, tenant_id),
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        return deleted

    def count(self, tenant_id: str) -> int:
        """Count episodic memories for a tenant."""
        conn = sqlite3.connect(self.db_path)
        count = conn.execute(
            "SELECT COUNT(*) FROM episodic_memories WHERE tenant_id = ?",
            (tenant_id,),
        ).fetchone()[0]
        conn.close()
        return count

    def count_unconsolidated(self, tenant_id: str) -> int:
        """Count unconsolidated episodic memories."""
        conn = sqlite3.connect(self.db_path)
        count = conn.execute(
            "SELECT COUNT(*) FROM episodic_memories WHERE tenant_id = ? AND is_consolidated = 0",
            (tenant_id,),
        ).fetchone()[0]
        conn.close()
        return count

    def get_oldest_unconsolidated(self, tenant_id: str, limit: int = 50) -> list[EpisodicMemory]:
        """Get oldest unconsolidated memories for consolidation."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT * FROM episodic_memories
            WHERE tenant_id = ? AND is_consolidated = 0
            ORDER BY timestamp ASC LIMIT ?
        """, (tenant_id, limit)).fetchall()
        conn.close()
        return [self._row_to_memory(r) for r in rows]

    def mark_consolidated(self, memory_ids: list[str]):
        """Mark memories as consolidated."""
        if not memory_ids:
            return
        conn = sqlite3.connect(self.db_path)
        placeholders = ",".join(["?" for _ in memory_ids])
        conn.execute(
            f"UPDATE episodic_memories SET is_consolidated = 1 WHERE memory_id IN ({placeholders})",
            memory_ids,
        )
        conn.commit()
        conn.close()

    def prune_low_importance(self, tenant_id: str, threshold: float = 0.2) -> int:
        """Delete low-importance consolidated memories. Returns count deleted."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            DELETE FROM episodic_memories
            WHERE tenant_id = ? AND is_consolidated = 1 AND importance_score < ?
        """, (tenant_id, threshold))
        conn.commit()
        pruned = cursor.rowcount
        conn.close()
        return pruned

    def get_sessions(self, tenant_id: str) -> list[str]:
        """Get all unique session IDs for a tenant."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT DISTINCT session_id FROM episodic_memories WHERE tenant_id = ? ORDER BY timestamp DESC",
            (tenant_id,),
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]

    def _row_to_memory(self, row: tuple) -> EpisodicMemory:
        meta = json.loads(row[8]) if row[8] else {}
        return EpisodicMemory(
            memory_id=row[0], tenant_id=row[1], session_id=row[2],
            timestamp=row[3], content=row[4], role=row[5],
            importance_score=row[6], is_consolidated=bool(row[7]),
            metadata=meta,
        )
