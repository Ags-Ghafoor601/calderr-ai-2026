"""
Enterprise AI Memory Platform — Procedural Memory Store
=========================================================
SQLite-backed procedural rule store with per-tenant isolation,
domain-based indexing, rule consolidation, and confidence promotion.
"""

import json
import sqlite3
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import ProceduralRule, ProceduralRuleCreate, ProceduralQueryRequest, RuleDomain


class ProceduralStore:
    """Per-tenant procedural rule store backed by SQLite."""

    def __init__(self, db_path: str = "data/memory_platform.db"):
        self.db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self):
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS procedural_rules (
                rule_id         TEXT PRIMARY KEY,
                tenant_id       TEXT NOT NULL,
                timestamp       TEXT NOT NULL,
                original_mistake TEXT NOT NULL,
                correction      TEXT NOT NULL,
                rule_text       TEXT NOT NULL,
                domain          TEXT NOT NULL DEFAULT 'general',
                confidence      REAL NOT NULL DEFAULT 0.7,
                application_count INTEGER NOT NULL DEFAULT 0,
                last_applied    TEXT,
                is_active       INTEGER NOT NULL DEFAULT 1
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pr_tenant ON procedural_rules(tenant_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pr_domain ON procedural_rules(tenant_id, domain)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pr_confidence ON procedural_rules(tenant_id, confidence DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_pr_active ON procedural_rules(tenant_id, is_active)")
        conn.commit()
        conn.close()

    def store(self, tenant_id: str, rule: ProceduralRuleCreate) -> ProceduralRule:
        """Store a new procedural rule."""
        entry = ProceduralRule(
            rule_id=str(uuid.uuid4())[:12],
            tenant_id=tenant_id,
            original_mistake=rule.original_mistake,
            correction=rule.correction,
            rule_text=rule.rule_text,
            domain=rule.domain,
            confidence=rule.confidence,
        )
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT INTO procedural_rules
            (rule_id, tenant_id, timestamp, original_mistake, correction,
             rule_text, domain, confidence, application_count, last_applied, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.rule_id, entry.tenant_id, entry.timestamp,
            entry.original_mistake, entry.correction, entry.rule_text,
            entry.domain.value, entry.confidence, 0, None, 1,
        ))
        conn.commit()
        conn.close()
        return entry

    def query(self, tenant_id: str, request: ProceduralQueryRequest) -> list[ProceduralRule]:
        """Query procedural rules for a tenant."""
        conn = sqlite3.connect(self.db_path)
        sql = "SELECT * FROM procedural_rules WHERE tenant_id = ?"
        params: list = [tenant_id]

        if request.active_only:
            sql += " AND is_active = 1"
        if request.domain:
            sql += " AND domain = ?"
            params.append(request.domain.value)

        sql += " ORDER BY confidence DESC, application_count DESC LIMIT ?"
        params.append(request.limit)

        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [self._row_to_rule(r) for r in rows]

    def get_all(self, tenant_id: str, active_only: bool = True) -> list[ProceduralRule]:
        """Get all rules for a tenant."""
        conn = sqlite3.connect(self.db_path)
        sql = "SELECT * FROM procedural_rules WHERE tenant_id = ?"
        params: list = [tenant_id]
        if active_only:
            sql += " AND is_active = 1"
        sql += " ORDER BY confidence DESC"
        rows = conn.execute(sql, params).fetchall()
        conn.close()
        return [self._row_to_rule(r) for r in rows]

    def increment_application(self, tenant_id: str, rule_id: str):
        """Increment application count and update last_applied."""
        now = datetime.now(timezone.utc).isoformat()
        conn = sqlite3.connect(self.db_path)
        conn.execute(
            "UPDATE procedural_rules SET application_count = application_count + 1, last_applied = ? "
            "WHERE rule_id = ? AND tenant_id = ?",
            (now, rule_id, tenant_id),
        )
        conn.commit()
        conn.close()

    def promote_rules(self, tenant_id: str, min_applications: int = 3, boost: float = 0.1) -> int:
        """Promote high-usage rules by boosting their confidence. Returns count promoted."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute("""
            UPDATE procedural_rules
            SET confidence = MIN(1.0, confidence + ?)
            WHERE tenant_id = ? AND is_active = 1
              AND application_count >= ? AND confidence < 1.0
        """, (boost, tenant_id, min_applications))
        conn.commit()
        promoted = cursor.rowcount
        conn.close()
        return promoted

    def consolidate(self, tenant_id: str) -> int:
        """Merge similar rules. Returns count of merges."""
        rules = self.get_all(tenant_id, active_only=True)
        merged = 0

        # Group by domain
        domain_groups: dict[str, list[ProceduralRule]] = {}
        for rule in rules:
            domain_groups.setdefault(rule.domain.value, []).append(rule)

        conn = sqlite3.connect(self.db_path)
        for _domain, group in domain_groups.items():
            text_groups: dict[str, list[ProceduralRule]] = {}
            for rule in group:
                key = rule.rule_text.lower().strip()[:100]
                text_groups.setdefault(key, []).append(rule)

            for _text, similar in text_groups.items():
                if len(similar) >= 3:
                    similar.sort(key=lambda r: r.confidence, reverse=True)
                    primary = similar[0]
                    new_conf = min(1.0, primary.confidence + 0.1 * (len(similar) - 1))
                    total_apps = sum(r.application_count for r in similar)
                    conn.execute(
                        "UPDATE procedural_rules SET confidence = ?, application_count = ? WHERE rule_id = ?",
                        (new_conf, total_apps, primary.rule_id),
                    )
                    for rule in similar[1:]:
                        conn.execute(
                            "UPDATE procedural_rules SET is_active = 0 WHERE rule_id = ?",
                            (rule.rule_id,),
                        )
                        merged += 1

        conn.commit()
        conn.close()
        return merged

    def delete(self, tenant_id: str, rule_id: str) -> bool:
        """Delete a procedural rule."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.execute(
            "DELETE FROM procedural_rules WHERE rule_id = ? AND tenant_id = ?",
            (rule_id, tenant_id),
        )
        conn.commit()
        deleted = cursor.rowcount > 0
        conn.close()
        return deleted

    def count(self, tenant_id: str) -> int:
        """Count active rules for a tenant."""
        conn = sqlite3.connect(self.db_path)
        count = conn.execute(
            "SELECT COUNT(*) FROM procedural_rules WHERE tenant_id = ? AND is_active = 1",
            (tenant_id,),
        ).fetchone()[0]
        conn.close()
        return count

    def _row_to_rule(self, row: tuple) -> ProceduralRule:
        domain_val = row[6]
        try:
            domain = RuleDomain(domain_val)
        except ValueError:
            domain = RuleDomain.GENERAL
        return ProceduralRule(
            rule_id=row[0], tenant_id=row[1], timestamp=row[2],
            original_mistake=row[3], correction=row[4], rule_text=row[5],
            domain=domain, confidence=row[7],
            application_count=row[8], last_applied=row[9],
            is_active=bool(row[10]),
        )
