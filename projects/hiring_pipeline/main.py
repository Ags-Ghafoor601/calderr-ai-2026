#!/usr/bin/env python3
"""
CalderR Internship – Week 4, Project 4-P-A
=============================================
AI-Powered Hiring Pipeline — Production Project

WHAT THIS PROJECT BUILDS:
-------------------------
A complete end-to-end hiring workflow combining FastAPI and LangGraph:
  • Ingest resumes → Score against job description (LLM)
  • Bias detection node → Shortlist top candidates
  • Generate tailored interview questions → Human review (HITL)
  • Final decision → Audit logging in SQLite

ARCHITECTURE:
    ┌─────────────────┐
    │  ingest_resumes  │  ← Register 10 candidates + DB insert
    └────────┬────────┘
    ┌────────▼────────┐
    │ score_candidates │  ← LLM scores each against job desc
    └────────┬────────┘
    ┌────────▼────────┐
    │   bias_check     │  ← Detect education/age/name bias
    └────────┬────────┘
    ┌────────▼────────┐
    │   shortlist      │  ← Top N above threshold
    └────────┬────────┘
    ┌────────▼──────────────┐
    │ generate_questions     │  ← LLM: 3 questions per candidate
    └────────┬──────────────┘
    ┌────────▼────────┐
    │  human_review    │  ← HITL — interrupt for manager
    └────────┬────────┘
        ⏸️ INTERRUPT
    ┌────────▼────────┐
    │ apply_decisions  │  ← Apply hire/reject decisions
    └────────┬────────┘
    ┌────────▼────────┐
    │  final_audit     │  ← Summary audit log to SQLite
    └────────┬────────┘
             │
            END

Run:
    python projects/hiring_pipeline/main.py demo
    python projects/hiring_pipeline/main.py graph
    python projects/hiring_pipeline/main.py audit
    python projects/hiring_pipeline/main.py bias-report
    python projects/hiring_pipeline/main.py serve        # FastAPI server
"""

import io
import os
import sys
import json
import uuid
import time
from pathlib import Path

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.rule import Rule
from rich import box

PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent.parent
sys.path.insert(0, str(PROJECT_DIR))

load_dotenv(ROOT_DIR / ".env")

from langgraph.checkpoint.sqlite import SqliteSaver
import database as db
from workflow import build_hiring_graph, get_initial_state
from sample_data import SAMPLE_RESUMES, JOB_DESCRIPTIONS

console = Console()
app = typer.Typer(
    name="hiring-pipeline",
    help="🏢 AI-Powered Hiring Pipeline — Project 4-P-A",
    add_completion=False,
)

CHECKPOINT_DB = str(PROJECT_DIR / ".hiring_checkpoint.db")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

def display_graph_structure():
    """Display the hiring pipeline graph."""
    console.print()
    tree = Tree("🔷 [bold cyan]AI-Powered Hiring Pipeline[/]")

    n1 = tree.add("📥 [bold]ingest_resumes[/] — Register candidates in SQLite")
    n2 = n1.add("📊 [bold]score_candidates[/] — LLM scores against job description")
    n3 = n2.add("🔍 [bold]bias_check[/] — Detect education/age/name bias")
    n4 = n3.add("✂️  [bold]shortlist[/] — Top N candidates above threshold")
    n5 = n4.add("❓ [bold]generate_questions[/] — LLM interview questions")
    n6 = n5.add("👤 [bold]human_review[/] — Queue for manager review")
    n6_int = n6.add("⏸️  [bold red]INTERRUPT[/] (await human decisions)")
    n7 = n6_int.add("▶️  [bold]apply_decisions[/] — Hire / reject")
    n8 = n7.add("📝 [bold]final_audit[/] — Summary audit entry → END")

    console.print(Panel(tree, title="📐 Pipeline Architecture", border_style="blue"))

    config_table = Table(title="⚙️  Configuration", box=box.ROUNDED)
    config_table.add_column("Parameter", style="bold cyan")
    config_table.add_column("Value", style="bold white")
    config_table.add_row("LLM Model", "llama-3.1-8b-instant")
    config_table.add_row("Shortlist Threshold", "55.0 / 100")
    config_table.add_row("Max Shortlist Size", "5 candidates")
    config_table.add_row("Persistence", "SQLite (candidates, scores, audit)")
    config_table.add_row("Checkpointer", "SqliteSaver (HITL state)")
    config_table.add_row("Bias Detection", "Education prestige, age, name, consistency")
    config_table.add_row("Interview Questions", "3 per candidate (tech, behavioral, situational)")
    console.print(config_table)


def display_scores(state: dict):
    """Display candidate scores table."""
    scores = state.get("candidate_scores", [])
    candidates = state.get("candidates", [])
    shortlisted = state.get("shortlisted_ids", [])

    if not scores:
        return

    console.print()
    table = Table(
        title=f"📊 Candidate Scores ({len(scores)} candidates)",
        box=box.ROUNDED,
    )
    table.add_column("Candidate", style="bold", max_width=22)
    table.add_column("Overall", justify="center", width=8)
    table.add_column("Skills", justify="center", width=8)
    table.add_column("Exp", justify="center", width=8)
    table.add_column("Edu", justify="center", width=8)
    table.add_column("Status", width=14)

    # Sort by overall score descending
    scored = sorted(scores, key=lambda s: s.get("overall_score", 0), reverse=True)

    for s in scored:
        cid = s["candidate_id"]
        name = next((c["name"] for c in candidates if c["id"] == cid), cid)
        overall = s.get("overall_score", 0)

        if overall >= 75:
            score_str = f"[bold green]{overall:.0f}[/]"
        elif overall >= 55:
            score_str = f"[yellow]{overall:.0f}[/]"
        else:
            score_str = f"[red]{overall:.0f}[/]"

        status = "[bold green]✅ SHORTLISTED[/]" if cid in shortlisted else "[dim]Not selected[/]"

        table.add_row(
            name,
            score_str,
            f"{s.get('skills_match', 0):.0f}",
            f"{s.get('experience_match', 0):.0f}",
            f"{s.get('education_match', 0):.0f}",
            status,
        )

    console.print(table)


def display_bias_reports(state: dict):
    """Display bias detection results."""
    reports = state.get("bias_reports", [])
    candidates = state.get("candidates", [])

    if not reports:
        return

    # Only show reports with meaningful flags (not just name-bias which is universal)
    significant = [
        r for r in reports
        if any(f.get("category") != "name_bias" for f in r.get("flags", []))
    ]

    if not significant:
        console.print()
        console.print("[green]✅ No significant bias flags detected across all candidates.[/]")
        return

    console.print()
    table = Table(
        title=f"🔍 Bias Detection Report ({len(significant)} flagged)",
        box=box.ROUNDED,
    )
    table.add_column("Candidate", style="bold", max_width=22)
    table.add_column("Risk", width=8)
    table.add_column("Flags", max_width=50)
    table.add_column("Adj. Score", justify="center", width=10)

    for r in significant:
        cid = r["candidate_id"]
        name = next((c["name"] for c in candidates if c["id"] == cid), cid)
        risk = r.get("overall_risk", "low")
        risk_str = {
            "high": "[bold red]HIGH[/]",
            "medium": "[yellow]MEDIUM[/]",
            "low": "[green]LOW[/]",
        }.get(risk, risk)

        non_name_flags = [
            f for f in r.get("flags", [])
            if f.get("category") != "name_bias"
        ]
        flag_str = "; ".join(
            f"[{f.get('severity', 'low')}] {f.get('category', '')}: {f.get('description', '')[:60]}"
            for f in non_name_flags
        )[:100]

        table.add_row(name, risk_str, flag_str or "—", f"{r.get('adjusted_score', 0):.0f}")

    console.print(table)


def display_questions(state: dict):
    """Display generated interview questions."""
    questions = state.get("interview_questions", {})
    candidates = state.get("candidates", [])

    if not questions:
        return

    console.print()
    for cid, qs in questions.items():
        name = next((c["name"] for c in candidates if c["id"] == cid), cid)
        console.print(f"\n[bold cyan]❓ Interview Questions for {name}:[/]")

        for i, q in enumerate(qs[:3], 1):
            cat = q.get("category", "general").upper()
            color = {"TECHNICAL": "blue", "BEHAVIORAL": "yellow", "SITUATIONAL": "magenta"}.get(cat, "white")
            console.print(f"  [{color}]{cat}[/{color}]: {q.get('question', '')}")
            if q.get("follow_up"):
                console.print(f"    [dim]Follow-up: {q['follow_up']}[/]")


def display_decisions(state: dict):
    """Display final hiring decisions."""
    decisions = state.get("final_decisions", [])
    candidates = state.get("candidates", [])

    if not decisions:
        return

    console.print()
    table = Table(title="📋 Final Hiring Decisions", box=box.DOUBLE)
    table.add_column("Candidate", style="bold", max_width=22)
    table.add_column("Decision", width=10)
    table.add_column("Decided By", width=10)
    table.add_column("Rationale", max_width=40)

    for d in decisions:
        cid = d["candidate_id"]
        name = next((c["name"] for c in candidates if c["id"] == cid), cid)
        dec = d.get("decision", "")
        dec_str = "[bold green]✅ HIRE[/]" if dec == "hire" else "[red]❌ REJECT[/]"

        table.add_row(name, dec_str, d.get("decided_by", ""), d.get("rationale", "")[:40])

    console.print(table)


def display_pipeline_result(state: dict):
    """Display comprehensive pipeline results."""
    log = state.get("processing_log", [])
    if log:
        console.print()
        tree = Tree("📋 [bold]Pipeline Timeline[/]")
        for entry in log:
            if "✅" in entry or "HIRED" in entry:
                tree.add(f"[green]{entry}[/]")
            elif "❌" in entry or "REJECT" in entry:
                tree.add(f"[red]{entry}[/]")
            elif "⏸️" in entry:
                tree.add(f"[yellow]{entry}[/]")
            elif "BIAS" in entry or "🔍" in entry:
                tree.add(f"[magenta]{entry}[/]")
            else:
                tree.add(f"[cyan]{entry}[/]")
        console.print(tree)

    display_scores(state)
    display_bias_reports(state)
    display_questions(state)
    display_decisions(state)


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------

@app.command()
def demo():
    """Run a full demo with 10 sample resumes against a job description."""
    console.print(Panel(
        "[bold cyan]Project 4-P-A — AI-Powered Hiring Pipeline[/]\n"
        "[dim]Production LangGraph workflow with bias detection and HITL[/]\n\n"
        "This demo processes 10 candidates for 'Senior Python Engineer':\n"
        "  1️⃣  Ingest all resumes into SQLite\n"
        "  2️⃣  LLM scores each candidate against job requirements\n"
        "  3️⃣  Bias detection analyses all scores\n"
        "  4️⃣  Shortlist top candidates (score ≥ 55)\n"
        "  5️⃣  Generate tailored interview questions\n"
        "  6️⃣  Human review (simulated: top 2 hired)\n"
        "  7️⃣  Final decisions + complete audit trail",
        title="🔬 Demo Mode",
        border_style="blue",
        padding=(1, 2),
    ))

    display_graph_structure()

    # Initialise database
    db_path = str(PROJECT_DIR / "hiring_pipeline.db")
    db.init_db(db_path)

    # Use the first job description
    job = JOB_DESCRIPTIONS[0]
    db.insert_job(job, db_path)

    console.print()
    console.print(Rule("🚀 Running Hiring Pipeline", style="bold green"))
    console.print(f"[cyan]Job:[/] {job['title']} ({job['id']})")
    console.print(f"[cyan]Candidates:[/] {len(SAMPLE_RESUMES)}")
    console.print(f"[cyan]Required Skills:[/] {', '.join(job['required_skills'])}")

    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        compiled = build_hiring_graph(checkpointer=checkpointer)
        thread_id = f"demo-{str(uuid.uuid4())[:6]}"
        config = {"configurable": {"thread_id": thread_id}}

        initial = get_initial_state(job, SAMPLE_RESUMES, thread_id, db_path)
        result = compiled.invoke(initial, config)

        # Pipeline should interrupt before apply_decisions
        if result.get("awaiting_human", False):
            console.print()
            console.print(Panel(
                "[bold yellow]⏸️  PIPELINE INTERRUPTED — awaiting human decisions[/]\n"
                "[dim]State persisted to SQLite checkpoint.[/]\n\n"
                "[bold green]▶️  Simulating hiring manager decisions:[/]",
                border_style="yellow",
            ))

            # Simulate human decisions: hire top 2, reject the rest
            shortlisted = result.get("shortlisted_ids", [])
            scores = result.get("candidate_scores", [])
            candidates = result.get("candidates", [])

            # Sort shortlisted by score
            scored = sorted(
                [(cid, next((s["overall_score"] for s in scores if s["candidate_id"] == cid), 0))
                 for cid in shortlisted],
                key=lambda x: x[1], reverse=True,
            )

            human_decisions = {}
            for i, (cid, score) in enumerate(scored):
                name = next((c["name"] for c in candidates if c["id"] == cid), cid)
                if i < 2:  # Top 2 hired
                    human_decisions[cid] = {
                        "decision": "hire",
                        "notes": f"Strong candidate — score {score:.0f}/100. Approved by hiring manager.",
                    }
                    console.print(f"  [green]✅ HIRE[/]: {name} (score: {score:.0f})")
                else:
                    human_decisions[cid] = {
                        "decision": "reject",
                        "notes": f"Good candidate but position filled — score {score:.0f}/100.",
                    }
                    console.print(f"  [red]❌ REJECT[/]: {name} (score: {score:.0f})")

            # Resume pipeline with human decisions
            compiled.update_state(config, {"human_decisions": human_decisions})
            result = compiled.invoke(None, config)

        display_pipeline_result(result)

    # Final summary
    console.print()
    total = len(SAMPLE_RESUMES)
    shortlisted = len(result.get("shortlisted_ids", []))
    hired = sum(1 for d in result.get("final_decisions", []) if d["decision"] == "hire")

    console.print(Panel(
        f"[bold green]✅ Pipeline complete![/]\n\n"
        f"📊 [cyan]{total}[/] candidates ingested\n"
        f"✂️  [cyan]{shortlisted}[/] shortlisted (threshold: 55)\n"
        f"✅ [cyan]{hired}[/] hired\n"
        f"❌ [cyan]{shortlisted - hired}[/] rejected at review\n\n"
        f"Key features demonstrated:\n"
        f"  • [cyan]LLM-powered scoring[/] against job requirements\n"
        f"  • [cyan]Bias detection[/] flags education prestige, age, name bias\n"
        f"  • [cyan]Human-in-the-loop[/] with SqliteSaver persistence\n"
        f"  • [cyan]Complete audit trail[/] in SQLite database\n"
        f"  • [cyan]Interview question generation[/] tailored per candidate",
        title="📊 Pipeline Summary",
        border_style="green",
    ))


@app.command(name="graph")
def graph_cmd():
    """Display the pipeline graph structure."""
    console.print(Rule("📐 AI-Powered Hiring Pipeline — Architecture", style="bold blue"))
    display_graph_structure()


@app.command()
def audit():
    """Display the complete audit trail."""
    console.print(Rule("📝 Audit Trail", style="bold blue"))

    db_path = str(PROJECT_DIR / "hiring_pipeline.db")
    if not Path(db_path).exists():
        console.print("[yellow]No database found. Run 'demo' first.[/]")
        raise typer.Exit()

    entries = db.get_audit_log(db_path=db_path)
    if not entries:
        console.print("[yellow]No audit entries found.[/]")
        raise typer.Exit()

    table = Table(
        title=f"📝 Audit Log ({len(entries)} entries)",
        box=box.ROUNDED,
    )
    table.add_column("Timestamp", style="dim", width=20)
    table.add_column("Action", style="bold cyan", width=18)
    table.add_column("Candidate", width=12)
    table.add_column("Details", max_width=50)
    table.add_column("By", width=8)
    table.add_column("Node", style="dim", width=18)

    for e in entries:
        table.add_row(
            e.get("timestamp", ""),
            e.get("action", ""),
            e.get("candidate_id", "")[:10],
            (e.get("details", "")[:48] + "…") if len(e.get("details", "")) > 48 else e.get("details", ""),
            e.get("decision_by", ""),
            e.get("node_name", ""),
        )

    console.print(table)


@app.command(name="bias-report")
def bias_report_cmd():
    """Display the bias detection report."""
    console.print(Rule("🔍 Bias Detection Report", style="bold magenta"))

    db_path = str(PROJECT_DIR / "hiring_pipeline.db")
    if not Path(db_path).exists():
        console.print("[yellow]No database found. Run 'demo' first.[/]")
        raise typer.Exit()

    from database import get_connection
    with get_connection(db_path) as conn:
        rows = conn.execute(
            "SELECT br.*, c.name, cs.overall_score AS original_score "
            "FROM bias_reports br "
            "LEFT JOIN candidates c ON br.candidate_id = c.id "
            "LEFT JOIN candidate_scores cs ON br.candidate_id = cs.candidate_id "
            "    AND br.job_id = cs.job_id "
            "ORDER BY br.overall_risk DESC, br.created_at"
        ).fetchall()

    if not rows:
        console.print("[yellow]No bias reports found. Run 'demo' first.[/]")
        raise typer.Exit()

    table = Table(title=f"🔍 Bias Analysis ({len(rows)} candidates)", box=box.ROUNDED)
    table.add_column("Candidate", style="bold", max_width=22)
    table.add_column("Risk", width=8)
    table.add_column("Original Score", justify="center", width=10)
    table.add_column("Adjusted", justify="center", width=10)
    table.add_column("# Flags", justify="center", width=8)
    table.add_column("Notes", max_width=40)

    for r in rows:
        d = dict(r)
        flags = json.loads(d.get("flags", "[]"))
        risk = d.get("overall_risk", "low")
        risk_str = {
            "high": "[bold red]HIGH[/]",
            "medium": "[yellow]MEDIUM[/]",
            "low": "[green]LOW[/]",
        }.get(risk, risk)

        non_name = [f for f in flags if f.get("category") != "name_bias"]
        original = d.get("original_score") or d.get("adjusted_score", 0)

        table.add_row(
            d.get("name", ""),
            risk_str,
            f"{original:.0f}",
            f"{d.get('adjusted_score', 0):.0f}",
            str(len(non_name)),
            (d.get("notes", "")[:38] + "…") if len(d.get("notes", "")) > 38 else d.get("notes", ""),
        )

    console.print(table)


@app.command()
def serve(
    port: int = typer.Option(8000, "--port", "-p", help="Port number"),
):
    """Start the FastAPI server."""
    console.print(Rule("🌐 Starting FastAPI Server", style="bold green"))

    try:
        import uvicorn
        from api import create_app
        api_app = create_app()
        console.print(f"[green]Server starting on http://localhost:{port}[/]")
        console.print("[dim]Press Ctrl+C to stop[/]")
        uvicorn.run(api_app, host="0.0.0.0", port=port)
    except ImportError:
        console.print("[yellow]FastAPI/uvicorn not available. Install with: pip install fastapi uvicorn[/]")
        raise typer.Exit(1)


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app()
