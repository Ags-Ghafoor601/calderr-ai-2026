#!/usr/bin/env python3
"""
CalderR Internship – Week 4, Lab 4.2
======================================
Self-Correcting Agent Loop — LangGraph Cyclic Workflows with Bounded Retries

WHAT THIS LAB BUILDS:
---------------------
A self-correcting agent that uses LangGraph's looping capability to:
  • Generate a response to a user query using an LLM
  • Validate the response against quality criteria (factual checks, length, format)
  • If validation fails: regenerate with feedback (up to 3 retries)
  • If validation passes: deliver the final polished response
  • Track and display iteration statistics

WHAT THIS TEACHES YOU:
----------------------
  • Building cyclic (looping) graphs in LangGraph
  • Conditional edges for loop-or-exit decisions
  • Proper termination conditions (max iterations)
  • State management with retry counters and feedback history
  • Debugging loops with step-by-step execution tracing

ARCHITECTURE:
              ┌────────────────┐
              │  receive_query │
              └───────┬────────┘
                      │
              ┌───────▼────────┐◄─────────────────┐
              │    generate    │                   │
              └───────┬────────┘                   │
                      │                            │
              ┌───────▼────────┐                   │
              │    validate    │                   │
              └───────┬────────┘                   │
                      │                            │
              ┌───────▼────────┐       ┌──────────┴──────┐
              │  pass or fail? │──────►│   regenerate    │
              └───────┬────────┘ fail  │ (with feedback) │
                      │ pass           └─────────────────┘
              ┌───────▼────────┐
              │    respond     │
              └───────┬────────┘
                      │
                     END

Run:
    python labs/lab_4_2_self_correcting_agent.py demo
    python labs/lab_4_2_self_correcting_agent.py query "Explain quantum computing"
    python labs/lab_4_2_self_correcting_agent.py stats
"""

import io
import os
import sys
import json
import time
from pathlib import Path
from typing import Optional, Annotated
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
from rich.live import Live
from rich.text import Text
from rich import box

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

console = Console()
app = typer.Typer(
    name="self-correcting",
    help="🔄 Self-Correcting Agent Loop — Lab 4.2",
    add_completion=False,
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
MAX_RETRIES = 3
STATS_FILE = ROOT_DIR / "labs" / "lab_4_2_stats.json"


# ---------------------------------------------------------------------------
# LLM Setup
# ---------------------------------------------------------------------------

def get_llm():
    """Create a ChatGroq LLM instance."""
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0.7,
        api_key=GROQ_API_KEY,
    )


# ---------------------------------------------------------------------------
# State Schema
# ---------------------------------------------------------------------------

class AgentState(TypedDict):
    """State for the self-correcting agent loop.

    Fields:
      - query: The user's original question
      - current_draft: The latest generated response
      - validation_passed: Whether the latest draft passed all checks
      - validation_feedback: Specific feedback on what failed
      - iteration: Current iteration number (0-indexed)
      - max_iterations: Maximum allowed iterations
      - history: List of all drafts and their validation results
      - processing_log: Accumulated log messages
      - final_response: The accepted response (set when validation passes)
    """
    query: str
    current_draft: str
    validation_passed: bool
    validation_feedback: str
    iteration: int
    max_iterations: int
    history: Annotated[list[dict], add]
    processing_log: Annotated[list[str], add]
    final_response: str


# ---------------------------------------------------------------------------
# Validation Criteria
# ---------------------------------------------------------------------------

def validate_response_quality(query: str, response: str, llm) -> tuple[bool, str]:
    """Validate response quality using multiple criteria.

    Checks:
      1. Minimum length (at least 100 characters)
      2. Relevance to the query
      3. Contains substantive content (not filler)
      4. Proper structure (sentences, not fragments)
      5. LLM-based quality check for factual correctness
    """
    issues = []

    # Check 1: Length
    if len(response.strip()) < 100:
        issues.append(f"Response too short ({len(response.strip())} chars, minimum 100)")

    # Check 2: Empty or whitespace
    if not response.strip():
        issues.append("Response is empty")
        return False, "; ".join(issues)

    # Check 3: Sentence structure
    sentences = [s.strip() for s in response.split('.') if s.strip()]
    if len(sentences) < 2:
        issues.append("Response lacks proper sentence structure (need at least 2 sentences)")

    # Check 4: Repetition detection
    words = response.lower().split()
    if len(words) > 10:
        unique_ratio = len(set(words)) / len(words)
        if unique_ratio < 0.3:
            issues.append(f"Excessive word repetition (unique ratio: {unique_ratio:.2f})")

    # Check 5: LLM-based quality validation
    try:
        validation_prompt = f"""You are a strict quality validator. Evaluate this response to the query.

Query: {query}

Response to evaluate:
{response}

Check for:
1. Is the response relevant to the query? 
2. Does it contain accurate, substantive information?
3. Is it well-structured and coherent?

Reply with EXACTLY one of:
- "PASS" if the response meets all criteria
- "FAIL: <specific reason>" if it does not

Your verdict:"""

        result = llm.invoke([HumanMessage(content=validation_prompt)])
        verdict = result.content.strip()

        if verdict.upper().startswith("FAIL"):
            reason = verdict.split(":", 1)[1].strip() if ":" in verdict else "Quality check failed"
            issues.append(f"LLM quality check: {reason}")
    except Exception as e:
        # Don't fail on validation errors — just log them
        issues.append(f"LLM validation error (non-blocking): {str(e)[:80]}")

    passed = len(issues) == 0
    feedback = "; ".join(issues) if issues else "All checks passed"
    return passed, feedback


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

def receive_query(state: AgentState) -> dict:
    """Initialize the pipeline with the user's query."""
    return {
        "iteration": 0,
        "processing_log": [
            f"[RECEIVE] Query received: \"{state['query'][:80]}{'...' if len(state['query']) > 80 else ''}\""
        ],
    }


def generate_response(state: AgentState) -> dict:
    """Generate or regenerate a response using the LLM."""
    iteration = state["iteration"]
    query = state["query"]
    feedback = state.get("validation_feedback", "")

    llm = get_llm()

    # Build prompt with feedback from previous iterations
    if iteration == 0:
        prompt = f"""Answer the following question thoroughly and accurately.
Provide a well-structured response with clear explanations.

Question: {query}

Your response:"""
    else:
        prompt = f"""Your previous answer was rejected for the following reason:
{feedback}

Please rewrite your answer to address these issues. Be more thorough, accurate, and detailed.

Original question: {query}

Your improved response:"""

    try:
        result = llm.invoke([
            SystemMessage(content="You are a knowledgeable assistant. Provide thorough, accurate, well-structured responses. Always give substantive answers with specific details."),
            HumanMessage(content=prompt),
        ])
        draft = result.content.strip()
    except Exception as e:
        draft = f"Error generating response: {str(e)}"

    return {
        "current_draft": draft,
        "processing_log": [
            f"[GENERATE] Iteration {iteration + 1}/{state['max_iterations']} — "
            f"Generated {len(draft):,} chars"
        ],
    }


def validate_response(state: AgentState) -> dict:
    """Validate the current draft against quality criteria."""
    llm = get_llm()
    passed, feedback = validate_response_quality(
        state["query"],
        state["current_draft"],
        llm,
    )

    status = "✅ PASSED" if passed else "❌ FAILED"
    return {
        "validation_passed": passed,
        "validation_feedback": feedback,
        "history": [{
            "iteration": state["iteration"] + 1,
            "draft": state["current_draft"],
            "passed": passed,
            "feedback": feedback,
            "draft_length": len(state["current_draft"]),
        }],
        "processing_log": [
            f"[VALIDATE] {status} — {feedback[:100]}{'...' if len(feedback) > 100 else ''}"
        ],
    }


def regenerate_response(state: AgentState) -> dict:
    """Prepare for regeneration by incrementing the iteration counter."""
    new_iteration = state["iteration"] + 1
    return {
        "iteration": new_iteration,
        "processing_log": [
            f"[REGENERATE] Preparing retry {new_iteration + 1}/{state['max_iterations']} — "
            f"Feedback: {state['validation_feedback'][:80]}..."
        ],
    }


def respond(state: AgentState) -> dict:
    """Deliver the final accepted response."""
    return {
        "final_response": state["current_draft"],
        "processing_log": [
            f"[RESPOND] ✅ Final response accepted after {state['iteration'] + 1} iteration(s) — "
            f"{len(state['current_draft']):,} chars"
        ],
    }


def respond_max_retries(state: AgentState) -> dict:
    """Handle max retries reached — deliver best available response."""
    # Find the longest draft from history as the best attempt
    history = state.get("history", [])
    best_draft = state["current_draft"]
    if history:
        best = max(history, key=lambda h: h.get("draft_length", 0))
        best_draft = best.get("draft", state["current_draft"])

    return {
        "final_response": best_draft,
        "processing_log": [
            f"[MAX-RETRIES] ⚠️ Maximum iterations ({state['max_iterations']}) reached — "
            f"Delivering best available response ({len(best_draft):,} chars)"
        ],
    }


# ---------------------------------------------------------------------------
# Conditional Edge Functions
# ---------------------------------------------------------------------------

def route_after_validation(state: AgentState) -> str:
    """Decide whether to respond, regenerate, or give up."""
    if state.get("validation_passed", False):
        return "respond"

    # Check if we've exceeded max iterations
    if state["iteration"] + 1 >= state["max_iterations"]:
        return "respond_max_retries"

    return "regenerate"


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the self-correcting agent graph."""
    graph = StateGraph(AgentState)

    # Add nodes
    graph.add_node("receive_query", receive_query)
    graph.add_node("generate", generate_response)
    graph.add_node("validate", validate_response)
    graph.add_node("regenerate", regenerate_response)
    graph.add_node("respond", respond)
    graph.add_node("respond_max_retries", respond_max_retries)

    # Set entry point
    graph.set_entry_point("receive_query")

    # Edges
    graph.add_edge("receive_query", "generate")
    graph.add_edge("generate", "validate")

    # Conditional: pass → respond, fail → regenerate or max-retries
    graph.add_conditional_edges(
        "validate",
        route_after_validation,
        {
            "respond": "respond",
            "regenerate": "regenerate",
            "respond_max_retries": "respond_max_retries",
        },
    )

    # Regenerate loops back to generate
    graph.add_edge("regenerate", "generate")
    graph.add_edge("respond", END)
    graph.add_edge("respond_max_retries", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Display Helpers
# ---------------------------------------------------------------------------

def display_result(result: dict):
    """Display the agent's processing result."""
    console.print()

    # Processing log timeline
    log = result.get("processing_log", [])
    if log:
        tree = Tree("📋 [bold]Execution Timeline[/]")
        for entry in log:
            if "✅" in entry or "RESPOND" in entry:
                tree.add(f"[green]{entry}[/]")
            elif "❌" in entry or "MAX-RETRIES" in entry:
                tree.add(f"[red]{entry}[/]")
            elif "REGENERATE" in entry:
                tree.add(f"[yellow]{entry}[/]")
            elif "GENERATE" in entry:
                tree.add(f"[cyan]{entry}[/]")
            else:
                tree.add(f"[white]{entry}[/]")
        console.print(tree)

    # Iteration history table
    history = result.get("history", [])
    if history:
        console.print()
        table = Table(
            title=f"🔄 Iteration History ({len(history)} attempt{'s' if len(history) != 1 else ''})",
            box=box.ROUNDED,
        )
        table.add_column("#", style="bold", width=3)
        table.add_column("Status", width=8)
        table.add_column("Draft Length", justify="right", style="cyan")
        table.add_column("Feedback", style="dim")

        for h in history:
            status = "[green]✅ PASS[/]" if h["passed"] else "[red]❌ FAIL[/]"
            fb = h["feedback"][:60] + ("..." if len(h["feedback"]) > 60 else "")
            table.add_row(str(h["iteration"]), status, f"{h['draft_length']:,}", fb)

        console.print(table)

    # Final response
    final = result.get("final_response", "")
    if final:
        console.print()
        border = "green" if result.get("validation_passed", False) else "yellow"
        label = "✅ Accepted Response" if result.get("validation_passed") else "⚠️ Best Available (max retries)"
        console.print(Panel(
            final,
            title=f"📝 {label}",
            border_style=border,
            padding=(1, 2),
        ))


def display_graph_structure():
    """Display the self-correcting agent graph structure."""
    console.print()
    tree = Tree("🔷 [bold cyan]Self-Correcting Agent Graph[/]")

    recv = tree.add("📩 [bold]receive_query[/] — Initialize pipeline")
    gen = recv.add("🤖 [bold]generate[/] — LLM generates response")
    val = gen.add("🔍 [bold]validate[/] — Quality checks (length, relevance, LLM-based)")
    cond = val.add("⚡ [bold yellow]CONDITIONAL EDGE[/]")

    cond.add("✅ [green]validation_passed = True[/] → [bold]respond[/] → END")
    retry = cond.add("❌ [yellow]validation_passed = False & retries left[/]")
    regen = retry.add("🔄 [bold]regenerate[/] (increment counter)")
    regen.add("↩️  [cyan]Loop back to [bold]generate[/][/]")
    cond.add("🛑 [red]max_iterations reached[/] → [bold]respond_max_retries[/] → END")

    console.print(Panel(tree, title="📐 Graph Architecture", border_style="blue"))

    config = Table(title="⚙️  Configuration", box=box.ROUNDED)
    config.add_column("Parameter", style="bold cyan")
    config.add_column("Value", style="bold white")
    config.add_row("LLM Model", "llama-3.1-8b-instant")
    config.add_row("Max Retries", str(MAX_RETRIES))
    config.add_row("Min Response Length", "100 chars")
    config.add_row("Validation", "Length + Structure + LLM Quality Check")
    console.print(config)


def save_stats(query: str, result: dict):
    """Save iteration statistics to a JSON file."""
    stats = []
    if STATS_FILE.exists():
        try:
            stats = json.loads(STATS_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, FileNotFoundError):
            stats = []

    history = result.get("history", [])
    stats.append({
        "query": query,
        "iterations": len(history),
        "passed": result.get("validation_passed", False),
        "final_length": len(result.get("final_response", "")),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    })

    STATS_FILE.write_text(json.dumps(stats, indent=2), encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------

@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask the agent"),
    max_retries: int = typer.Option(MAX_RETRIES, "--max-retries", "-r", help="Max retry attempts"),
):
    """Ask the self-correcting agent a question."""
    console.print(Rule("🔄 Self-Correcting Agent — Lab 4.2", style="bold blue"))
    display_graph_structure()

    console.print()
    console.print(Rule("🚀 Executing Pipeline", style="bold green"))
    console.print(f"[cyan]Query:[/] {question}")
    console.print(f"[dim]Max iterations: {max_retries}[/]")

    compiled = build_graph()
    result = compiled.invoke({
        "query": question,
        "current_draft": "",
        "validation_passed": False,
        "validation_feedback": "",
        "iteration": 0,
        "max_iterations": max_retries,
        "history": [],
        "processing_log": [],
        "final_response": "",
    })

    display_result(result)
    save_stats(question, result)
    console.print(f"\n[dim]📊 Stats saved to {STATS_FILE.name}[/]")


@app.command()
def stats():
    """Display iteration statistics from previous runs."""
    console.print(Rule("📊 Iteration Statistics — Lab 4.2", style="bold blue"))

    if not STATS_FILE.exists():
        console.print("[yellow]No statistics yet. Run some queries first![/]")
        raise typer.Exit()

    data = json.loads(STATS_FILE.read_text(encoding="utf-8"))
    if not data:
        console.print("[yellow]No statistics yet.[/]")
        raise typer.Exit()

    table = Table(title=f"📊 Agent Performance ({len(data)} runs)", box=box.ROUNDED)
    table.add_column("#", style="dim", width=3)
    table.add_column("Query", style="cyan", max_width=40)
    table.add_column("Iterations", justify="center")
    table.add_column("Result", width=8)
    table.add_column("Response Length", justify="right")
    table.add_column("Timestamp", style="dim")

    for i, entry in enumerate(data, 1):
        q = entry["query"][:38] + ("…" if len(entry["query"]) > 38 else "")
        status = "[green]✅ Pass[/]" if entry["passed"] else "[yellow]⚠️ Max[/]"
        table.add_row(
            str(i), q, str(entry["iterations"]),
            status, f"{entry['final_length']:,}",
            entry.get("timestamp", ""),
        )

    console.print(table)

    # Summary stats
    total = len(data)
    passed = sum(1 for d in data if d["passed"])
    avg_iter = sum(d["iterations"] for d in data) / total
    console.print()
    summary = Table(title="📈 Summary", box=box.SIMPLE)
    summary.add_column("Metric", style="bold cyan")
    summary.add_column("Value", style="bold white")
    summary.add_row("Total Runs", str(total))
    summary.add_row("First-Try Pass Rate", f"{sum(1 for d in data if d['iterations'] == 1 and d['passed'])}/{total}")
    summary.add_row("Overall Pass Rate", f"{passed}/{total} ({passed/total*100:.0f}%)")
    summary.add_row("Average Iterations", f"{avg_iter:.1f}")
    console.print(summary)


@app.command()
def demo():
    """Run a full demonstration with multiple test queries."""
    console.print(Panel(
        "[bold cyan]Lab 4.2 — Self-Correcting Agent Loop[/]\n"
        "[dim]LangGraph cyclic workflow with bounded retries[/]\n\n"
        "This demo shows the agent:\n"
        "  1️⃣  Generates a response to a query\n"
        "  2️⃣  Validates the response (length, relevance, LLM quality check)\n"
        "  3️⃣  If validation fails: regenerates with feedback (up to 3 retries)\n"
        "  4️⃣  If validation passes: delivers the final response\n"
        "  5️⃣  Tracks iteration statistics across all queries",
        title="🔬 Demo Mode",
        border_style="blue",
        padding=(1, 2),
    ))

    display_graph_structure()

    test_queries = [
        "What are the three main benefits of using LangGraph over simple LangChain chains?",
        "Explain the concept of human-in-the-loop in AI agent systems.",
        "What is a TypedDict state schema and why is it important in LangGraph?",
    ]

    compiled = build_graph()

    for i, q in enumerate(test_queries, 1):
        console.print()
        console.print(Rule(f"🧪 Test {i}/{len(test_queries)}: {q[:60]}...", style="bold cyan"))

        result = compiled.invoke({
            "query": q,
            "current_draft": "",
            "validation_passed": False,
            "validation_feedback": "",
            "iteration": 0,
            "max_iterations": MAX_RETRIES,
            "history": [],
            "processing_log": [],
            "final_response": "",
        })

        display_result(result)
        save_stats(q, result)

    # Final summary
    console.print()
    console.print(Panel(
        "[bold green]✅ Demo complete![/]\n\n"
        "Key takeaways:\n"
        "  • [cyan]Looping graphs[/] enable iterative self-improvement\n"
        "  • [cyan]Bounded retries[/] prevent infinite loops\n"
        "  • [cyan]Validation feedback[/] guides the regeneration\n"
        "  • [cyan]Statistics tracking[/] enables performance analysis",
        title="📊 Demo Summary",
        border_style="green",
    ))


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app()
