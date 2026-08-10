"""
Procedural Memory & Self-Improving Agent — Memory Store
========================================================
SQLite-backed procedural memory store for correction rules,
interaction logs, and performance tracking.
"""

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from models import CorrectionRule, InteractionLog, PerformanceRecord, RuleDomain


class ProceduralMemoryStore:
    """SQLite-backed store for correction rules, interaction logs, and performance records."""

    def __init__(self, db_path: str = "self_improving_agent.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Initialise all database tables."""
        conn = sqlite3.connect(self.db_path)

        # Correction rules table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS correction_rules (
                rule_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                original_mistake TEXT NOT NULL,
                correction TEXT NOT NULL,
                rule_text TEXT NOT NULL,
                domain TEXT NOT NULL DEFAULT 'general',
                confidence REAL NOT NULL DEFAULT 0.7,
                application_count INTEGER NOT NULL DEFAULT 0,
                last_applied TEXT,
                source_interaction_id TEXT DEFAULT '',
                is_active INTEGER NOT NULL DEFAULT 1
            )
        """)

        # Interaction logs table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS interaction_logs (
                interaction_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                user_message TEXT NOT NULL,
                agent_response TEXT NOT NULL,
                was_corrected INTEGER NOT NULL DEFAULT 0,
                correction_text TEXT,
                rules_applied TEXT DEFAULT '[]',
                quality_score REAL
            )
        """)

        # Performance records table
        conn.execute("""
            CREATE TABLE IF NOT EXISTS performance_records (
                record_id TEXT PRIMARY KEY,
                timestamp TEXT NOT NULL,
                interaction_number INTEGER NOT NULL,
                was_correct INTEGER NOT NULL DEFAULT 1,
                error_type TEXT,
                rules_applied_count INTEGER DEFAULT 0,
                total_rules_available INTEGER DEFAULT 0,
                quality_score REAL DEFAULT 0.5
            )
        """)

        # Indexes
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rules_domain ON correction_rules(domain)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rules_confidence ON correction_rules(confidence DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_rules_active ON correction_rules(is_active)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_logs_timestamp ON interaction_logs(timestamp DESC)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_perf_number ON performance_records(interaction_number)")

        conn.commit()
        conn.close()

    # ─── Rule Operations ──────────────────────────────────────────

    def store_rule(self, rule: CorrectionRule):
        """Store a correction rule."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO correction_rules
            (rule_id, timestamp, original_mistake, correction, rule_text,
             domain, confidence, application_count, last_applied,
             source_interaction_id, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            rule.rule_id, rule.timestamp, rule.original_mistake,
            rule.correction, rule.rule_text, rule.domain.value,
            rule.confidence, rule.application_count, rule.last_applied,
            rule.source_interaction_id, 1 if rule.is_active else 0,
        ))
        conn.commit()
        conn.close()

    def get_all_rules(self, active_only: bool = True) -> list[CorrectionRule]:
        """Retrieve all correction rules."""
        conn = sqlite3.connect(self.db_path)
        query = "SELECT * FROM correction_rules"
        if active_only:
            query += " WHERE is_active = 1"
        query += " ORDER BY confidence DESC, application_count DESC"
        rows = conn.execute(query).fetchall()
        conn.close()
        return [self._row_to_rule(r) for r in rows]

    def get_rules_by_domain(self, domain: str) -> list[CorrectionRule]:
        """Retrieve rules for a specific domain."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT * FROM correction_rules WHERE domain = ? AND is_active = 1 ORDER BY confidence DESC",
            (domain,),
        ).fetchall()
        conn.close()
        return [self._row_to_rule(r) for r in rows]

    def increment_rule_application(self, rule_id: str):
        """Increment the application count of a rule and update last_applied."""
        conn = sqlite3.connect(self.db_path)
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "UPDATE correction_rules SET application_count = application_count + 1, last_applied = ? WHERE rule_id = ?",
            (now, rule_id),
        )
        conn.commit()
        conn.close()

    def consolidate_similar_rules(self) -> int:
        """Merge similar rules into higher-confidence single rules. Returns count of merges."""
        rules = self.get_all_rules()
        merged_count = 0

        # Group by domain
        domain_groups: dict[str, list[CorrectionRule]] = {}
        for rule in rules:
            domain_groups.setdefault(rule.domain.value, []).append(rule)

        conn = sqlite3.connect(self.db_path)

        for _domain, group in domain_groups.items():
            # Simple similarity: rules with identical rule_text (case-insensitive)
            text_groups: dict[str, list[CorrectionRule]] = {}
            for rule in group:
                key = rule.rule_text.lower().strip()
                text_groups.setdefault(key, []).append(rule)

            for _text, similar in text_groups.items():
                if len(similar) >= 3:  # Consolidate when 3+ similar rules exist
                    # Keep the one with highest confidence, merge others
                    similar.sort(key=lambda r: r.confidence, reverse=True)
                    primary = similar[0]

                    # Boost confidence
                    new_confidence = min(1.0, primary.confidence + 0.1 * (len(similar) - 1))
                    total_applications = sum(r.application_count for r in similar)

                    conn.execute(
                        "UPDATE correction_rules SET confidence = ?, application_count = ? WHERE rule_id = ?",
                        (new_confidence, total_applications, primary.rule_id),
                    )

                    # Deactivate merged rules
                    for rule in similar[1:]:
                        conn.execute(
                            "UPDATE correction_rules SET is_active = 0 WHERE rule_id = ?",
                            (rule.rule_id,),
                        )
                        merged_count += 1

        conn.commit()
        conn.close()
        return merged_count

    def count_rules(self) -> int:
        """Count active rules."""
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM correction_rules WHERE is_active = 1").fetchone()[0]
        conn.close()
        return count

    def _row_to_rule(self, row: tuple) -> CorrectionRule:
        """Convert a database row to a CorrectionRule object."""
        return CorrectionRule(
            rule_id=row[0], timestamp=row[1], original_mistake=row[2],
            correction=row[3], rule_text=row[4],
            domain=RuleDomain(row[5]) if row[5] in [e.value for e in RuleDomain] else RuleDomain.GENERAL,
            confidence=row[6], application_count=row[7],
            last_applied=row[8], source_interaction_id=row[9],
            is_active=bool(row[10]),
        )

    # ─── Interaction Log Operations ───────────────────────────────

    def log_interaction(self, log: InteractionLog):
        """Store an interaction log."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO interaction_logs
            (interaction_id, timestamp, user_message, agent_response,
             was_corrected, correction_text, rules_applied, quality_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            log.interaction_id, log.timestamp, log.user_message,
            log.agent_response, 1 if log.was_corrected else 0,
            log.correction_text, json.dumps(log.rules_applied),
            log.quality_score,
        ))
        conn.commit()
        conn.close()

    def get_recent_interactions(self, limit: int = 20) -> list[InteractionLog]:
        """Retrieve recent interactions."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT * FROM interaction_logs ORDER BY timestamp DESC LIMIT ?",
            (limit,),
        ).fetchall()
        conn.close()
        return [
            InteractionLog(
                interaction_id=r[0], timestamp=r[1], user_message=r[2],
                agent_response=r[3], was_corrected=bool(r[4]),
                correction_text=r[5],
                rules_applied=json.loads(r[6]) if r[6] else [],
                quality_score=r[7],
            )
            for r in rows
        ]

    def count_interactions(self) -> int:
        """Count total interactions."""
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM interaction_logs").fetchone()[0]
        conn.close()
        return count

    def count_corrections(self) -> int:
        """Count interactions where the agent was corrected."""
        conn = sqlite3.connect(self.db_path)
        count = conn.execute("SELECT COUNT(*) FROM interaction_logs WHERE was_corrected = 1").fetchone()[0]
        conn.close()
        return count

    # ─── Performance Record Operations ────────────────────────────

    def record_performance(self, record: PerformanceRecord):
        """Store a performance record."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO performance_records
            (record_id, timestamp, interaction_number, was_correct,
             error_type, rules_applied_count, total_rules_available, quality_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            record.record_id, record.timestamp, record.interaction_number,
            1 if record.was_correct else 0, record.error_type,
            record.rules_applied_count, record.total_rules_available,
            record.quality_score,
        ))
        conn.commit()
        conn.close()

    def get_performance_records(self) -> list[PerformanceRecord]:
        """Retrieve all performance records ordered by interaction number."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT * FROM performance_records ORDER BY interaction_number ASC"
        ).fetchall()
        conn.close()
        return [
            PerformanceRecord(
                record_id=r[0], timestamp=r[1], interaction_number=r[2],
                was_correct=bool(r[3]), error_type=r[4],
                rules_applied_count=r[5], total_rules_available=r[6],
                quality_score=r[7],
            )
            for r in rows
        ]

    def get_error_rate(self, window: int = 5) -> float:
        """Get the error rate over the last N interactions."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute(
            "SELECT was_correct FROM performance_records ORDER BY interaction_number DESC LIMIT ?",
            (window,),
        ).fetchall()
        conn.close()
        if not rows:
            return 0.0
        errors = sum(1 for r in rows if not r[0])
        return errors / len(rows)
