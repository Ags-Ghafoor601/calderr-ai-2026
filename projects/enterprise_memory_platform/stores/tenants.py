"""
Enterprise AI Memory Platform — Tenant Manager
=================================================
Manages tenants (users/organisations) and provides
statistics across all memory stores.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import Tenant, TenantStats, PlatformStats


class TenantManager:
    """Manages tenants and provides cross-store statistics."""

    def __init__(self, db_path: str = "data/memory_platform.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS tenants (
                tenant_id   TEXT PRIMARY KEY,
                name        TEXT NOT NULL,
                created_at  TEXT NOT NULL,
                is_active   INTEGER NOT NULL DEFAULT 1,
                metadata    TEXT DEFAULT '{}'
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_tenants_active ON tenants(is_active)")
        conn.commit()
        conn.close()

    def create_tenant(self, tenant_id: str, name: str, metadata: Optional[dict] = None) -> Tenant:
        """Create a new tenant."""
        tenant = Tenant(
            tenant_id=tenant_id.strip().lower(),
            name=name,
            metadata=metadata or {},
        )
        conn = sqlite3.connect(self.db_path)
        try:
            conn.execute(
                "INSERT INTO tenants (tenant_id, name, created_at, is_active, metadata) VALUES (?, ?, ?, ?, ?)",
                (tenant.tenant_id, tenant.name, tenant.created_at, 1, json.dumps(tenant.metadata)),
            )
            conn.commit()
        except sqlite3.IntegrityError:
            # Tenant already exists — update
            conn.execute(
                "UPDATE tenants SET name = ?, is_active = 1, metadata = ? WHERE tenant_id = ?",
                (tenant.name, json.dumps(tenant.metadata), tenant.tenant_id),
            )
            conn.commit()
        conn.close()
        return tenant

    def get_tenant(self, tenant_id: str) -> Optional[Tenant]:
        """Get a tenant by ID."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT * FROM tenants WHERE tenant_id = ?", (tenant_id.lower(),)
        ).fetchone()
        conn.close()
        if row:
            return Tenant(
                tenant_id=row[0], name=row[1], created_at=row[2],
                is_active=bool(row[3]),
                metadata=json.loads(row[4]) if row[4] else {},
            )
        return None

    def list_tenants(self, active_only: bool = True) -> list[Tenant]:
        """List all tenants."""
        conn = sqlite3.connect(self.db_path)
        sql = "SELECT * FROM tenants"
        if active_only:
            sql += " WHERE is_active = 1"
        sql += " ORDER BY created_at DESC"
        rows = conn.execute(sql).fetchall()
        conn.close()
        return [
            Tenant(
                tenant_id=r[0], name=r[1], created_at=r[2],
                is_active=bool(r[3]),
                metadata=json.loads(r[4]) if r[4] else {},
            )
            for r in rows
        ]

    def deactivate_tenant(self, tenant_id: str) -> bool:
        """Deactivate a tenant."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "UPDATE tenants SET is_active = 0 WHERE tenant_id = ?",
            (tenant_id.lower(),),
        )
        conn.commit()
        updated = cursor.rowcount > 0
        conn.close()
        return updated

    def tenant_exists(self, tenant_id: str) -> bool:
        """Check if a tenant exists and is active."""
        conn = sqlite3.connect(self.db_path)
        row = conn.execute(
            "SELECT is_active FROM tenants WHERE tenant_id = ?",
            (tenant_id.lower(),),
        ).fetchone()
        conn.close()
        return bool(row and row[0])

    def count_tenants(self, active_only: bool = True) -> int:
        """Count tenants."""
        conn = sqlite3.connect(self.db_path)
        sql = "SELECT COUNT(*) FROM tenants"
        if active_only:
            sql += " WHERE is_active = 1"
        count = conn.execute(sql).fetchone()[0]
        conn.close()
        return count
