#!/usr/bin/env python3
"""
CalderR Internship – Week 4, Lab 4.3
======================================
Human-in-the-Loop Approval Workflow — Interrupt, Persist, Resume

WHAT THIS LAB BUILDS:
---------------------
A content moderation graph that:
  • Receives posts and classifies them (safe / borderline / harmful)
  • Auto-approves safe content, auto-rejects harmful content
  • Routes borderline content to a human reviewer (interrupt)
  • Persists state across the interrupt using SqliteSaver
  • Resumes execution after human decision
  • Logs all moderation decisions with timestamps

WHAT THIS TEACHES YOU:
----------------------
  • LangGraph interrupt patterns for human-in-the-loop
  • SqliteSaver checkpointing for persistent state
  • Resuming interrupted graphs from exact checkpoint
  • Conditional routing with three+ branches
  • Real-world content moderation workflow design

ARCHITECTURE:
             ┌────────────┐
             │  receive    │
             │   post      │
             └──────┬──────┘
                    │
             ┌──────▼──────┐
             │  classify    │
             │  content     │
             └──────┬──────┘
                    │
        ┌───────────┼───────────┐
        │           │           │
   ┌────▼────┐ ┌───▼────┐ ┌───▼─────┐
   │  auto   │ │ human  │ │  auto   │
   │ approve │ │ review │ │ reject  │
   └────┬────┘ └───┬────┘ └───┬─────┘
        │    INTERRUPT│         │
        │     (pause) │         │
        │    ┌───▼────┐        │
        │    │ apply   │        │
        │    │ decision│        │
        │    └───┬────┘        │
        │        │              │
        └────────┼──────────────┘
           ┌─────▼─────┐
           │   log      │
           │ decision   │
           └─────┬─────┘
                 │
                END

Run:
    python labs/lab_4_3_human_in_the_loop.py demo
    python labs/lab_4_3_human_in_the_loop.py moderate "Your content here"
    python labs/lab_4_3_human_in_the_loop.py pending
    python labs/lab_4_3_human_in_the_loop.py review <thread-id> approve|reject
    python labs/lab_4_3_human_in_the_loop.py history
    python labs/lab_4_3_human_in_the_loop.py graph
"""

import io
import os
import sys
import json
import time
import uuid
import sqlite3
from pathlib import Path
from typing import Optional, Annotated, Literal
from operator import add

# Fix Windows console encoding for Rich Unicode output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import typer
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.rule import Rule
from rich import box

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

console = Console()
app = typer.Typer(
    name="content-moderation",
    help="🛡️ Content Moderation HITL — Lab 4.3",
    add_completion=False,
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHECKPOINT_DB = str(ROOT_DIR / "labs" / ".checkpoint_lab43.db")
DECISION_LOG = ROOT_DIR / "labs" / "lab_4_3_decisions.json"


# ---------------------------------------------------------------------------
# LLM Setup
# ---------------------------------------------------------------------------

def get_llm():
    """Create a ChatGroq LLM instance."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.1,
        api_key=GROQ_API_KEY,
    )


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------

class ModerationState(TypedDict):
    """State for content moderation workflow.

    Fields:
      - post_content: The content submitted for moderation
      - post_id: Unique identifier for the post
      - classification: safe / borderline / harmful
      - confidence: Classifier confidence (0.0 - 1.0)
      - classification_reason: Why the content was classified this way
      - human_decision: approve / reject (set by human reviewer)
      - human_notes: Optional notes from human reviewer
      - final_decision: The final moderation outcome
      - decision_source: auto / human
      - processing_log: Accumulated log entries
      - awaiting_human: Whether the graph is paused for human input
    """
    post_content: str
    post_id: str
    classification: str
    confidence: float
    classification_reason: str
    human_decision: str
    human_notes: str
    final_decision: str
    decision_source: str
    processing_log: Annotated[list[str], add]
    awaiting_human: bool


# ---------------------------------------------------------------------------
# Content Classification
# ---------------------------------------------------------------------------

HARMFUL_KEYWORDS = [
    "hack", "exploit", "attack", "malware", "phishing",
    "steal", "illegal", "scam", "fraud", "weapon",
]

BORDERLINE_KEYWORDS = [
    "controversial", "debate", "opinion", "political", "sensitive",
    "criticism", "dispute", "argue", "conflict", "bias",
]


def classify_content(content: str, llm) -> tuple[str, float, str]:
    """Classify content as safe, borderline, or harmful.

    Uses a combination of keyword detection and LLM-based classification.
    Returns (classification, confidence, reason).
    """
    content_lower = content.lower()

    # Quick keyword checks
    harmful_matches = [kw for kw in HARMFUL_KEYWORDS if kw in content_lower]
    borderline_matches = [kw for kw in BORDERLINE_KEYWORDS if kw in content_lower]

    # If strongly harmful keywords detected, fast-track
    if len(harmful_matches) >= 2:
        return (
            "harmful",
            0.95,
            f"Multiple harmful keywords detected: {', '.join(harmful_matches)}",
        )

    # LLM-based classification for nuanced content
    try:
        prompt = f"""You are a content moderator. Classify the following content into one of three categories:

Content: "{content}"

Categories:
1. SAFE - Normal, appropriate content with no issues
2. BORDERLINE - Content that may need human review (controversial opinions, sensitive topics, ambiguous intent)
3. HARMFUL - Content that promotes violence, illegal activities, scams, or abuse

Respond with EXACTLY this JSON format (no markdown, no code blocks):
{{"classification": "safe|borderline|harmful", "confidence": 0.0-1.0, "reason": "brief explanation"}}

Your classification:"""

        result = llm.invoke([HumanMessage(content=prompt)])
        response_text = result.content.strip()

        # Clean up response - remove markdown code blocks if present
        if response_text.startswith("```"):
            lines = response_text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            response_text = "\n".join(lines).strip()

        parsed = json.loads(response_text)
        classification = parsed.get("classification", "borderline").lower()
        confidence = float(parsed.get("confidence", 0.5))
        reason = parsed.get("reason", "LLM classification")

        # Validate classification value
        if classification not in ("safe", "borderline", "harmful"):
            classification = "borderline"

        # Override with keyword evidence if present
        if harmful_matches and classification == "safe":
            classification = "borderline"
            reason += f" (harmful keywords also detected: {', '.join(harmful_matches)})"

        return classification, confidence, reason

    except Exception as e:
        # Fallback: use keyword-based classification
        if harmful_matches:
            return "borderline", 0.6, f"Keyword-based (LLM error): harmful keywords: {', '.join(harmful_matches)}"
        if borderline_matches:
            return "borderline", 0.5, f"Keyword-based (LLM error): borderline keywords: {', '.join(borderline_matches)}"
        return "safe", 0.4, f"Default safe (LLM error: {str(e)[:50]})"


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

def receive_post(state: ModerationState) -> dict:
    """Register the incoming post for moderation."""
    post_id = state.get("post_id") or str(uuid.uuid4())[:8]
    content = state["post_content"]
    return {
        "post_id": post_id,
        "processing_log": [
            f"[RECEIVE] Post {post_id} received — "
            f"{len(content)} chars, {len(content.split())} words"
        ],
    }


def classify_post(state: ModerationState) -> dict:
    """Classify the post content."""
    llm = get_llm()
    classification, confidence, reason = classify_content(state["post_content"], llm)

    emoji = {"safe": "🟢", "borderline": "🟡", "harmful": "🔴"}[classification]
    return {
        "classification": classification,
        "confidence": confidence,
        "classification_reason": reason,
        "processing_log": [
            f"[CLASSIFY] {emoji} {classification.upper()} "
            f"(confidence: {confidence:.0%}) — {reason}"
        ],
    }


def auto_approve(state: ModerationState) -> dict:
    """Automatically approve safe content."""
    return {
        "final_decision": "approved",
        "decision_source": "auto",
        "processing_log": [
            f"[AUTO-APPROVE] ✅ Post {state['post_id']} auto-approved — "
            f"Content classified as safe ({state['confidence']:.0%} confidence)"
        ],
    }


def auto_reject(state: ModerationState) -> dict:
    """Automatically reject harmful content."""
    return {
        "final_decision": "rejected",
        "decision_source": "auto",
        "processing_log": [
            f"[AUTO-REJECT] ❌ Post {state['post_id']} auto-rejected — "
            f"Content classified as harmful ({state['confidence']:.0%} confidence)"
        ],
    }


def human_review(state: ModerationState) -> dict:
    """Mark the post as awaiting human review — this is where we'll interrupt."""
    return {
        "awaiting_human": True,
        "processing_log": [
            f"[HUMAN-REVIEW] ⏸️  Post {state['post_id']} queued for human review — "
            f"Classification: {state['classification']} ({state['confidence']:.0%}), "
            f"Reason: {state['classification_reason']}"
        ],
    }


def apply_human_decision(state: ModerationState) -> dict:
    """Apply the human reviewer's decision."""
    decision = state.get("human_decision", "reject")
    notes = state.get("human_notes", "No notes provided")

    final = "approved" if decision.lower() in ("approve", "approved") else "rejected"
    return {
        "final_decision": final,
        "decision_source": "human",
        "awaiting_human": False,
        "processing_log": [
            f"[HUMAN-DECISION] {'✅' if final == 'approved' else '❌'} "
            f"Post {state['post_id']} {final} by human reviewer — "
            f"Notes: {notes}"
        ],
    }


def log_decision(state: ModerationState) -> dict:
    """Log the final moderation decision for audit."""
    # Save to decision log file
    log_entry = {
        "post_id": state["post_id"],
        "content_preview": state["post_content"][:100],
        "classification": state["classification"],
        "confidence": state["confidence"],
        "final_decision": state["final_decision"],
        "decision_source": state["decision_source"],
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    existing = []
    if DECISION_LOG.exists():
        try:
            existing = json.loads(DECISION_LOG.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            existing = []

    existing.append(log_entry)
    DECISION_LOG.write_text(json.dumps(existing, indent=2), encoding="utf-8")

    return {
        "processing_log": [
            f"[LOG] 📝 Decision logged — Post {state['post_id']}: "
            f"{state['final_decision']} (by {state['decision_source']})"
        ],
    }


# ---------------------------------------------------------------------------
# Conditional Edge
# ---------------------------------------------------------------------------

def route_after_classification(state: ModerationState) -> str:
    """Route based on content classification."""
    classification = state.get("classification", "borderline")
    if classification == "safe":
        return "auto_approve"
    elif classification == "harmful":
        return "auto_reject"
    else:
        return "human_review"


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_graph(checkpointer=None):
    """Build the content moderation graph.

    When a checkpointer is provided, the graph persists state across
    interrupts, enabling true human-in-the-loop workflows.
    """
    graph = StateGraph(ModerationState)

    # Add nodes
    graph.add_node("receive_post", receive_post)
    graph.add_node("classify_post", classify_post)
    graph.add_node("auto_approve", auto_approve)
    graph.add_node("auto_reject", auto_reject)
    graph.add_node("human_review", human_review)
    graph.add_node("apply_human_decision", apply_human_decision)
    graph.add_node("log_decision", log_decision)

    # Entry point
    graph.set_entry_point("receive_post")

    # Edges
    graph.add_edge("receive_post", "classify_post")

    # Three-way conditional routing
    graph.add_conditional_edges(
        "classify_post",
        route_after_classification,
        {
            "auto_approve": "auto_approve",
            "auto_reject": "auto_reject",
            "human_review": "human_review",
        },
    )

    # Auto paths go to log
    graph.add_edge("auto_approve", "log_decision")
    graph.add_edge("auto_reject", "log_decision")

    # Human review path: review → apply decision → log
    graph.add_edge("human_review", "apply_human_decision")
    graph.add_edge("apply_human_decision", "log_decision")

    # Log is the final node
    graph.add_edge("log_decision", END)

    # Compile with checkpointer and interrupt before apply_human_decision
    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["apply_human_decision"],
    )


def get_initial_state(content: str, post_id: str = "") -> dict:
    """Create initial state for a moderation run."""
    return {
        "post_content": content,
        "post_id": post_id or str(uuid.uuid4())[:8],
        "classification": "",
        "confidence": 0.0,
        "classification_reason": "",
        "human_decision": "",
        "human_notes": "",
        "final_decision": "",
        "decision_source": "",
        "processing_log": [],
        "awaiting_human": False,
    }


# ---------------------------------------------------------------------------
# Display Helpers
# ---------------------------------------------------------------------------

def display_result(result: dict, title: str = "Moderation Result"):
    """Display moderation result with rich formatting."""
    console.print()

    # Processing log
    log = result.get("processing_log", [])
    if log:
        tree = Tree("📋 [bold]Processing Timeline[/]")
        for entry in log:
            if "✅" in entry or "APPROVE" in entry:
                tree.add(f"[green]{entry}[/]")
            elif "❌" in entry or "REJECT" in entry:
                tree.add(f"[red]{entry}[/]")
            elif "⏸️" in entry or "HUMAN" in entry:
                tree.add(f"[yellow]{entry}[/]")
            elif "CLASSIFY" in entry:
                if "SAFE" in entry:
                    tree.add(f"[green]{entry}[/]")
                elif "HARMFUL" in entry:
                    tree.add(f"[red]{entry}[/]")
                else:
                    tree.add(f"[yellow]{entry}[/]")
            else:
                tree.add(f"[cyan]{entry}[/]")
        console.print(tree)

    # Result summary
    final = result.get("final_decision", "")
    if final:
        color = "green" if final == "approved" else "red"
        emoji = "✅" if final == "approved" else "❌"
        source = result.get("decision_source", "unknown")

        console.print()
        console.print(Panel(
            f"[bold {color}]{emoji} {final.upper()}[/]\n\n"
            f"[cyan]Post ID:[/] {result.get('post_id', 'N/A')}\n"
            f"[cyan]Classification:[/] {result.get('classification', 'N/A')} "
            f"({result.get('confidence', 0):.0%} confidence)\n"
            f"[cyan]Decision by:[/] {source}\n"
            f"[cyan]Reason:[/] {result.get('classification_reason', 'N/A')}",
            title=f"🛡️ {title}",
            border_style=color,
        ))
    elif result.get("awaiting_human"):
        console.print()
        console.print(Panel(
            f"[bold yellow]⏸️ AWAITING HUMAN REVIEW[/]\n\n"
            f"[cyan]Post ID:[/] {result.get('post_id', 'N/A')}\n"
            f"[cyan]Classification:[/] {result.get('classification', 'N/A')} "
            f"({result.get('confidence', 0):.0%} confidence)\n"
            f"[cyan]Reason:[/] {result.get('classification_reason', 'N/A')}\n\n"
            f"[dim]Content preview:[/] {result.get('post_content', '')[:150]}...",
            title="🛡️ Pending Human Review",
            border_style="yellow",
        ))


def display_graph_structure():
    """Display the content moderation graph."""
    console.print()
    tree = Tree("🔷 [bold cyan]Content Moderation Graph[/]")

    recv = tree.add("📩 [bold]receive_post[/] — Register incoming content")
    classify = recv.add("🔍 [bold]classify_post[/] — LLM + keyword classification")
    cond = classify.add("⚡ [bold yellow]THREE-WAY CONDITIONAL EDGE[/]")

    safe = cond.add("🟢 [green]safe[/] → [bold]auto_approve[/]")
    safe.add("📝 [bold]log_decision[/] → END")

    borderline = cond.add("🟡 [yellow]borderline[/] → [bold]human_review[/]")
    interrupt = borderline.add("⏸️  [bold red]INTERRUPT[/] (graph pauses here)")
    resume = interrupt.add("▶️  [bold]apply_human_decision[/] (resumes after human input)")
    resume.add("📝 [bold]log_decision[/] → END")

    harmful = cond.add("🔴 [red]harmful[/] → [bold]auto_reject[/]")
    harmful.add("📝 [bold]log_decision[/] → END")

    console.print(Panel(tree, title="📐 Graph Architecture", border_style="blue"))

    config = Table(title="⚙️  Configuration", box=box.ROUNDED)
    config.add_column("Parameter", style="bold cyan")
    config.add_column("Value", style="bold white")
    config.add_row("LLM Model", "llama-3.1-8b-instant")
    config.add_row("Checkpoint DB", Path(CHECKPOINT_DB).name)
    config.add_row("Decision Log", DECISION_LOG.name)
    config.add_row("Interrupt Point", "Before apply_human_decision")
    config.add_row("Persistence", "SqliteSaver (SQLite)")
    console.print(config)


# ---------------------------------------------------------------------------
# Sample Content
# ---------------------------------------------------------------------------

SAMPLE_POSTS = [
    {
        "label": "Safe Content (should auto-approve)",
        "content": "I just finished building my first LangGraph workflow! "
                   "It processes documents through a pipeline with conditional "
                   "routing based on document size. Really enjoying learning "
                   "about graph-based AI orchestration.",
        "color": "green",
    },
    {
        "label": "Borderline Content (should need human review)",
        "content": "This controversial political opinion piece argues that "
                   "current AI regulation is fundamentally biased against "
                   "open-source development. The debate around AI safety has "
                   "become increasingly politically sensitive, with both sides "
                   "presenting conflicting viewpoints.",
        "color": "yellow",
    },
    {
        "label": "Harmful Content (should auto-reject)",
        "content": "Here's how to hack into corporate email systems and exploit "
                   "security vulnerabilities for phishing attacks. This guide "
                   "covers malware deployment and how to steal credentials.",
        "color": "red",
    },
    {
        "label": "Safe Educational Content",
        "content": "Today I learned about embeddings in machine learning. "
                   "An embedding is a dense vector representation that captures "
                   "semantic meaning. Cosine similarity measures the angle between "
                   "two vectors to determine how semantically similar they are.",
        "color": "green",
    },
]


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------

@app.command()
def moderate(
    content: str = typer.Argument(..., help="Content to moderate"),
):
    """Submit content for moderation."""
    console.print(Rule("🛡️ Content Moderation — Lab 4.3", style="bold blue"))
    display_graph_structure()
    console.print()
    console.print(Rule("🚀 Processing", style="bold green"))

    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        compiled = build_graph(checkpointer=checkpointer)
        thread_id = str(uuid.uuid4())[:8]
        config = {"configurable": {"thread_id": thread_id}}

        result = compiled.invoke(get_initial_state(content, thread_id), config)
        display_result(result)

        # If awaiting human review, inform the user
        if result.get("awaiting_human"):
            console.print()
            console.print(Panel(
                f"To approve:  [bold cyan]python labs/lab_4_3_human_in_the_loop.py review {thread_id} approve[/]\n"
                f"To reject:   [bold cyan]python labs/lab_4_3_human_in_the_loop.py review {thread_id} reject[/]",
                title="📋 Next Steps",
                border_style="yellow",
            ))


@app.command()
def review(
    thread_id: str = typer.Argument(..., help="Thread ID of the pending post"),
    decision: str = typer.Argument(..., help="approve or reject"),
    notes: str = typer.Option("", "--notes", "-n", help="Optional reviewer notes"),
):
    """Review a pending post (human-in-the-loop)."""
    console.print(Rule(f"🛡️ Human Review — Thread {thread_id}", style="bold yellow"))

    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        compiled = build_graph(checkpointer=checkpointer)
        config = {"configurable": {"thread_id": thread_id}}

        # Get current state
        current_state = compiled.get_state(config)
        if not current_state or not current_state.values:
            console.print(f"[red]❌ No pending review found for thread {thread_id}[/]")
            raise typer.Exit(1)

        console.print(f"[cyan]Post content:[/] {current_state.values.get('post_content', '')[:150]}...")
        console.print(f"[cyan]Classification:[/] {current_state.values.get('classification', 'N/A')}")
        console.print(f"[cyan]Your decision:[/] [bold]{decision}[/]")
        console.print()

        # Update state with human decision and resume
        compiled.update_state(config, {
            "human_decision": decision,
            "human_notes": notes or f"Reviewed and {decision}d by human moderator",
        })

        # Resume execution
        result = compiled.invoke(None, config)
        display_result(result, "Human Review Complete")


@app.command()
def pending():
    """List all posts pending human review."""
    console.print(Rule("📋 Pending Reviews — Lab 4.3", style="bold yellow"))

    if not Path(CHECKPOINT_DB).exists():
        console.print("[yellow]No checkpoint database found. Submit content first.[/]")
        raise typer.Exit()

    # Check the decision log for pending items
    console.print("[dim]Check threads by looking at the moderation output for pending thread IDs.[/]")
    console.print("[dim]Use 'moderate' command to submit content, then 'review' to approve/reject.[/]")


@app.command()
def history():
    """Show moderation decision history."""
    console.print(Rule("📊 Decision History — Lab 4.3", style="bold blue"))

    if not DECISION_LOG.exists():
        console.print("[yellow]No decisions logged yet.[/]")
        raise typer.Exit()

    data = json.loads(DECISION_LOG.read_text(encoding="utf-8"))
    if not data:
        console.print("[yellow]No decisions logged yet.[/]")
        raise typer.Exit()

    table = Table(title=f"📊 Moderation History ({len(data)} decisions)", box=box.ROUNDED)
    table.add_column("Post ID", style="bold cyan", width=10)
    table.add_column("Classification", width=12)
    table.add_column("Decision", width=10)
    table.add_column("Source", width=8)
    table.add_column("Content Preview", max_width=40)
    table.add_column("Timestamp", style="dim")

    for entry in data:
        cls_color = {"safe": "green", "borderline": "yellow", "harmful": "red"}.get(entry["classification"], "white")
        dec_color = "green" if entry["final_decision"] == "approved" else "red"
        table.add_row(
            entry["post_id"],
            f"[{cls_color}]{entry['classification']}[/]",
            f"[{dec_color}]{entry['final_decision']}[/]",
            entry["decision_source"],
            entry["content_preview"][:38] + "…",
            entry["timestamp"],
        )

    console.print(table)

    # Summary
    approved = sum(1 for d in data if d["final_decision"] == "approved")
    rejected = len(data) - approved
    auto = sum(1 for d in data if d["decision_source"] == "auto")
    human = len(data) - auto
    console.print()
    summary = Table(title="📈 Summary", box=box.SIMPLE)
    summary.add_column("Metric", style="bold cyan")
    summary.add_column("Value", style="bold white")
    summary.add_row("Total Decisions", str(len(data)))
    summary.add_row("Approved / Rejected", f"{approved} / {rejected}")
    summary.add_row("Auto / Human", f"{auto} / {human}")
    console.print(summary)


@app.command()
def graph():
    """Display the graph structure."""
    console.print(Rule("📐 Content Moderation Graph — Lab 4.3", style="bold blue"))
    display_graph_structure()


@app.command()
def demo():
    """Run a full demonstration of the content moderation workflow."""
    console.print(Panel(
        "[bold cyan]Lab 4.3 — Human-in-the-Loop Approval Workflow[/]\n"
        "[dim]Content moderation with interrupt, persist, and resume[/]\n\n"
        "This demo shows:\n"
        "  1️⃣  [green]Safe content[/] → auto-approved\n"
        "  2️⃣  [yellow]Borderline content[/] → human review (simulated)\n"
        "  3️⃣  [red]Harmful content[/] → auto-rejected\n"
        "  4️⃣  [green]Safe educational content[/] → auto-approved\n\n"
        "[dim]State is persisted in SQLite between human review interrupts[/]",
        title="🔬 Demo Mode",
        border_style="blue",
        padding=(1, 2),
    ))

    display_graph_structure()

    with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
        compiled = build_graph(checkpointer=checkpointer)

        for i, sample in enumerate(SAMPLE_POSTS, 1):
            console.print()
            console.print(Rule(
                f"🧪 Test {i}/{len(SAMPLE_POSTS)}: {sample['label']}",
                style=f"bold {sample['color']}",
            ))
            console.print(f"[dim]{sample['content'][:120]}...[/]")

            thread_id = f"demo-{i}-{str(uuid.uuid4())[:4]}"
            config = {"configurable": {"thread_id": thread_id}}
            initial_state = get_initial_state(sample["content"], thread_id)

            result = compiled.invoke(initial_state, config)

            # If this post was routed to human review (interrupt triggered),
            # simulate a human decision and resume
            if result.get("awaiting_human", False):
                console.print()
                console.print(Panel(
                    "[bold yellow]⏸️  Graph INTERRUPTED — awaiting human review[/]\n"
                    "[dim]State has been persisted to SQLite checkpoint.[/]\n"
                    "[bold green]▶️  Simulating human reviewer: APPROVE[/]",
                    border_style="yellow",
                ))

                # Simulate human reviewer providing a decision
                compiled.update_state(config, {
                    "human_decision": "approve",
                    "human_notes": "Reviewed by human: content is acceptable despite borderline classification",
                })

                # Resume graph execution from checkpoint
                result = compiled.invoke(None, config)

            display_result(result)

    # Final summary
    console.print()
    console.print(Panel(
        "[bold green]✅ Demo complete![/]\n\n"
        "Key takeaways:\n"
        "  • [cyan]Three-way routing[/] classifies content into safe/borderline/harmful\n"
        "  • [cyan]Interrupt pattern[/] pauses execution for human review\n"
        "  • [cyan]SqliteSaver[/] persists state across the interrupt\n"
        "  • [cyan]Resume[/] continues from the exact checkpoint\n"
        "  • [cyan]Audit logging[/] records all decisions with timestamps",
        title="📊 Demo Summary",
        border_style="green",
    ))


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app()
