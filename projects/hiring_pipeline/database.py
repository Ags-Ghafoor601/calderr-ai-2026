"""
AI-Powered Hiring Pipeline — SQLite Database Layer
====================================================
Handles all persistence for candidates, scores, bias reports,
audit logs, and hiring decisions.
"""

import sqlite3
import json
import time
from pathlib import Path
from contextlib import contextmanager
from typing import Optional


DB_PATH = Path(__file__).resolve().parent / "hiring_pipeline.db"


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

@contextmanager
def get_connection(db_path: str = str(DB_PATH)):
    """Context manager for SQLite connections with WAL mode."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Schema initialization
# ---------------------------------------------------------------------------

def init_db(db_path: str = str(DB_PATH)):
    """Create all tables if they don't exist."""
    with get_connection(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS candidates (
                id              TEXT PRIMARY KEY,
                name            TEXT NOT NULL,
                email           TEXT NOT NULL,
                phone           TEXT DEFAULT '',
                years_experience INTEGER DEFAULT 0,
                education       TEXT DEFAULT '',
                university      TEXT DEFAULT '',
                skills          TEXT DEFAULT '[]',
                previous_roles  TEXT DEFAULT '[]',
                summary         TEXT DEFAULT '',
                resume_text     TEXT DEFAULT '',
                status          TEXT DEFAULT 'ingested',
                created_at      TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS job_descriptions (
                id                   TEXT PRIMARY KEY,
                title                TEXT NOT NULL,
                department           TEXT DEFAULT 'Engineering',
                required_skills      TEXT DEFAULT '[]',
                preferred_skills     TEXT DEFAULT '[]',
                min_experience       INTEGER DEFAULT 0,
                education_requirement TEXT DEFAULT '',
                description          TEXT DEFAULT '',
                salary_range         TEXT DEFAULT '',
                created_at           TEXT DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS candidate_scores (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id     TEXT NOT NULL,
                job_id           TEXT NOT NULL,
                overall_score    REAL DEFAULT 0.0,
                skills_match     REAL DEFAULT 0.0,
                experience_match REAL DEFAULT 0.0,
                education_match  REAL DEFAULT 0.0,
                summary_relevance REAL DEFAULT 0.0,
                scoring_rationale TEXT DEFAULT '',
                created_at       TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (candidate_id) REFERENCES candidates(id),
                FOREIGN KEY (job_id) REFERENCES job_descriptions(id)
            );

            CREATE TABLE IF NOT EXISTS bias_reports (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id    TEXT NOT NULL,
                job_id          TEXT NOT NULL,
                flags           TEXT DEFAULT '[]',
                overall_risk    TEXT DEFAULT 'low',
                adjusted_score  REAL DEFAULT 0.0,
                notes           TEXT DEFAULT '',
                created_at      TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (candidate_id) REFERENCES candidates(id)
            );

            CREATE TABLE IF NOT EXISTS audit_log (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp       TEXT DEFAULT (datetime('now')),
                candidate_id    TEXT DEFAULT '',
                job_id          TEXT DEFAULT '',
                action          TEXT NOT NULL,
                details         TEXT DEFAULT '',
                decision_by     TEXT DEFAULT 'system',
                node_name       TEXT DEFAULT ''
            );

            CREATE TABLE IF NOT EXISTS hiring_decisions (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                candidate_id     TEXT NOT NULL,
                job_id           TEXT NOT NULL,
                decision         TEXT NOT NULL,
                decided_by       TEXT DEFAULT 'auto',
                rationale        TEXT DEFAULT '',
                questions        TEXT DEFAULT '[]',
                bias_report      TEXT DEFAULT '{}',
                score_data       TEXT DEFAULT '{}',
                created_at       TEXT DEFAULT (datetime('now')),
                FOREIGN KEY (candidate_id) REFERENCES candidates(id)
            );
        """)


# ---------------------------------------------------------------------------
# Candidate CRUD
# ---------------------------------------------------------------------------

def insert_candidate(candidate: dict, db_path: str = str(DB_PATH)) -> str:
    """Insert a candidate and return their ID."""
    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO candidates
               (id, name, email, phone, years_experience, education,
                university, skills, previous_roles, summary, resume_text, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                candidate.get("id", ""),
                candidate.get("name", ""),
                candidate.get("email", ""),
                candidate.get("phone", ""),
                candidate.get("years_experience", 0),
                candidate.get("education", ""),
                candidate.get("university", ""),
                json.dumps(candidate.get("skills", [])),
                json.dumps(candidate.get("previous_roles", [])),
                candidate.get("summary", ""),
                candidate.get("resume_text", ""),
                candidate.get("status", "ingested"),
            ),
        )
    return candidate.get("id", "")


def update_candidate_status(candidate_id: str, status: str, db_path: str = str(DB_PATH)):
    """Update a candidate's status."""
    with get_connection(db_path) as conn:
        conn.execute(
            "UPDATE candidates SET status = ? WHERE id = ?",
            (status, candidate_id),
        )


def get_candidate(candidate_id: str, db_path: str = str(DB_PATH)) -> Optional[dict]:
    """Retrieve a candidate by ID."""
    with get_connection(db_path) as conn:
        row = conn.execute(
            "SELECT * FROM candidates WHERE id = ?", (candidate_id,)
        ).fetchone()
        if row:
            d = dict(row)
            d["skills"] = json.loads(d.get("skills", "[]"))
            d["previous_roles"] = json.loads(d.get("previous_roles", "[]"))
            return d
    return None


def get_all_candidates(db_path: str = str(DB_PATH)) -> list[dict]:
    """Retrieve all candidates."""
    with get_connection(db_path) as conn:
        rows = conn.execute("SELECT * FROM candidates ORDER BY created_at").fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["skills"] = json.loads(d.get("skills", "[]"))
            d["previous_roles"] = json.loads(d.get("previous_roles", "[]"))
            results.append(d)
        return results


# ---------------------------------------------------------------------------
# Job Description CRUD
# ---------------------------------------------------------------------------

def insert_job(job: dict, db_path: str = str(DB_PATH)) -> str:
    """Insert a job description."""
    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT OR REPLACE INTO job_descriptions
               (id, title, department, required_skills, preferred_skills,
                min_experience, education_requirement, description, salary_range)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                job.get("id", ""),
                job.get("title", ""),
                job.get("department", ""),
                json.dumps(job.get("required_skills", [])),
                json.dumps(job.get("preferred_skills", [])),
                job.get("min_experience", 0),
                job.get("education_requirement", ""),
                job.get("description", ""),
                job.get("salary_range", ""),
            ),
        )
    return job.get("id", "")


# ---------------------------------------------------------------------------
# Score CRUD
# ---------------------------------------------------------------------------

def insert_score(score: dict, job_id: str, db_path: str = str(DB_PATH)):
    """Insert a candidate score."""
    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT INTO candidate_scores
               (candidate_id, job_id, overall_score, skills_match,
                experience_match, education_match, summary_relevance,
                scoring_rationale)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                score.get("candidate_id", ""),
                job_id,
                score.get("overall_score", 0.0),
                score.get("skills_match", 0.0),
                score.get("experience_match", 0.0),
                score.get("education_match", 0.0),
                score.get("summary_relevance", 0.0),
                score.get("scoring_rationale", ""),
            ),
        )


# ---------------------------------------------------------------------------
# Bias Report CRUD
# ---------------------------------------------------------------------------

def insert_bias_report(report: dict, job_id: str, db_path: str = str(DB_PATH)):
    """Insert a bias report."""
    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT INTO bias_reports
               (candidate_id, job_id, flags, overall_risk, adjusted_score, notes)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                report.get("candidate_id", ""),
                job_id,
                json.dumps(report.get("flags", [])),
                report.get("overall_risk", "low"),
                report.get("adjusted_score", 0.0),
                report.get("notes", ""),
            ),
        )


# ---------------------------------------------------------------------------
# Audit Log
# ---------------------------------------------------------------------------

def log_audit(action: str, details: str, candidate_id: str = "",
              job_id: str = "", decision_by: str = "system",
              node_name: str = "", db_path: str = str(DB_PATH)):
    """Write an audit log entry."""
    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT INTO audit_log
               (timestamp, candidate_id, job_id, action, details,
                decision_by, node_name)
               VALUES (datetime('now'), ?, ?, ?, ?, ?, ?)""",
            (candidate_id, job_id, action, details, decision_by, node_name),
        )


def get_audit_log(candidate_id: str = "", db_path: str = str(DB_PATH)) -> list[dict]:
    """Retrieve audit log entries, optionally filtered by candidate."""
    with get_connection(db_path) as conn:
        if candidate_id:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE candidate_id = ? ORDER BY timestamp",
                (candidate_id,),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY timestamp"
            ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Hiring Decision
# ---------------------------------------------------------------------------

def insert_decision(decision: dict, job_id: str, db_path: str = str(DB_PATH)):
    """Record a hiring decision."""
    with get_connection(db_path) as conn:
        conn.execute(
            """INSERT INTO hiring_decisions
               (candidate_id, job_id, decision, decided_by, rationale,
                questions, bias_report, score_data)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                decision.get("candidate_id", ""),
                job_id,
                decision.get("decision", ""),
                decision.get("decided_by", "auto"),
                decision.get("rationale", ""),
                json.dumps(decision.get("questions", [])),
                json.dumps(decision.get("bias_report", {})),
                json.dumps(decision.get("score_data", {})),
            ),
        )


def get_all_decisions(db_path: str = str(DB_PATH)) -> list[dict]:
    """Retrieve all hiring decisions."""
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT * FROM hiring_decisions ORDER BY created_at"
        ).fetchall()
        results = []
        for row in rows:
            d = dict(row)
            d["questions"] = json.loads(d.get("questions", "[]"))
            d["bias_report"] = json.loads(d.get("bias_report", "{}"))
            d["score_data"] = json.loads(d.get("score_data", "{}"))
            results.append(d)
        return results
