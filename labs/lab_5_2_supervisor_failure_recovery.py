#!/usr/bin/env python3
"""
CalderR Internship – Week 5, Lab 5.2
======================================
Supervisor with Failure Recovery — Resilient Multi-Agent Delegation

WHAT THIS LAB BUILDS:
---------------------
A Supervisor Agent that:
  • Delegates tasks to 3 specialist agents (DataAgent, AnalyticsAgent, SummaryAgent)
  • Handles 2 injected failure modes: random timeout + low-confidence response
  • Detects failure type → logs reasoning → re-routes to alternative agent
  • If all alternatives fail → produces a gracefully degraded response
  • NEVER crashes, regardless of how many agents fail simultaneously
  • Logs every delegation decision with detailed reasoning

WHAT THIS TEACHES YOU:
----------------------
  • Agent failure is ROUTINE in production, not exceptional
  • Supervisor pattern: single point of coordination + decision logging
  • Failure detection strategies: timeout, confidence thresholds, error types
  • Graceful degradation: always produce a useful output, even partial
  • LangGraph StateGraph for multi-agent workflows with conditional routing

ARCHITECTURE:
    ┌──────────────────────┐
    │   SUPERVISOR AGENT   │ ← Receives complex task, plans delegation
    └──────────┬───────────┘
               │ delegates
    ┌──────────▼───────────────────────────────┐
    │         SPECIALIST POOL                   │
    │  ┌─────────┐ ┌──────────┐ ┌────────────┐ │
    │  │  Data   │ │Analytics │ │  Summary   │ │
    │  │  Agent  │ │  Agent   │ │   Agent    │ │
    │  └────┬────┘ └────┬─────┘ └─────┬──────┘ │
    │       │  FAILURES: │             │        │
    │       │ • Timeout  │             │        │
    │       │ • LowConf  │             │        │
    └───────┼────────────┼─────────────┼────────┘
            │            │             │
    ┌───────▼────────────▼─────────────▼────────┐
    │           FAILURE HANDLER                  │
    │  detect type → log reasoning →             │
    │  re-route OR graceful degradation          │
    └───────────────────┬───────────────────────┘
                        │
    ┌───────────────────▼───────────────────────┐
    │           RESULT AGGREGATOR                │
    │  merge partial + full results → final      │
    └───────────────────────────────────────────┘

Run:
    python labs/lab_5_2_supervisor_failure_recovery.py demo
    python labs/lab_5_2_supervisor_failure_recovery.py inject-timeout
    python labs/lab_5_2_supervisor_failure_recovery.py inject-low-confidence
    python labs/lab_5_2_supervisor_failure_recovery.py stress-test
"""

# pylint: disable=line-too-long, too-many-locals, wrong-import-position, duplicate-code
import io
import os
import sys
import time
import random
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional
from enum import Enum

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import typer
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.rule import Rule
from rich import box
from groq import Groq
from langgraph.graph import StateGraph, START, END

load_dotenv(ROOT_DIR / ".env")

console = Console()
app = typer.Typer(help="Lab 5.2 — Supervisor with Failure Recovery")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = "llama-3.1-8b-instant"

# ─── LLM Helper ────────────────────────────────────────────────────────────
def llm_call(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """Make a single LLM call via Groq with timeout handling and retry."""
    client = Groq(api_key=GROQ_API_KEY)
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=1024,
            )
            return resp.choices[0].message.content.strip()
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if "429" in str(exc) or "rate_limit" in str(exc).lower():
                wait = (attempt + 1) * 12
                console.print(f"    [dim]Rate limited, waiting {wait}s...[/]")
                time.sleep(wait)
            else:
                raise
    return "Unable to generate response after retries."


# ═══════════════════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════

class FailureMode(str, Enum):
    """Enumeration of possible failure modes."""
    NONE = "none"
    TIMEOUT = "timeout"
    LOW_CONFIDENCE = "low_confidence"
    EXCEPTION = "exception"


class AgentStatus(str, Enum):
    """Enumeration of agent statuses."""
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    DEGRADED = "degraded"


class DelegationDecision(BaseModel):
    """Log entry for every delegation decision the supervisor makes."""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    decision_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_name: str
    task_type: str
    action: str  # "delegate", "re-route", "fallback", "skip"
    reasoning: str
    failure_detected: Optional[str] = None
    alternative_agent: Optional[str] = None


class SpecialistResult(BaseModel):
    """Result from a specialist agent."""
    agent_name: str
    task_type: str
    output: str
    confidence: float = Field(ge=0.0, le=1.0)
    processing_time_ms: float
    status: AgentStatus
    failure_mode: FailureMode = FailureMode.NONE
    attempt_number: int = 1


class SupervisorState(BaseModel):
    """Complete state for the supervisor workflow."""
    task: str = ""
    subtasks: list[str] = Field(default_factory=list)
    delegation_log: list[dict] = Field(default_factory=list)
    specialist_results: list[dict] = Field(default_factory=list)
    failed_agents: list[str] = Field(default_factory=list)
    current_agent_idx: int = 0
    final_output: str = ""
    total_retries: int = 0
    system_status: str = "running"
    injected_failures: dict[str, str] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
#  SPECIALIST AGENTS
# ═══════════════════════════════════════════════════════════════════════════

CONFIDENCE_THRESHOLD = 0.5
SPECIALISTS = {
    "data-agent": {
        "name": "data-agent",
        "system_prompt": (
            "You are a data specialist. Given a topic, provide 3-5 key data points, "
            "statistics, or factual findings. Be precise and cite specific numbers where possible."
        ),
        "task_type": "data_gathering",
        "fallback": "analytics-agent",
    },
    "analytics-agent": {
        "name": "analytics-agent",
        "system_prompt": (
            "You are an analytics specialist. Analyse the given topic and provide: "
            "1) Trend analysis, 2) Key patterns, 3) Comparative insights. Be analytical and structured."
        ),
        "task_type": "analysis",
        "fallback": "summary-agent",
    },
    "summary-agent": {
        "name": "summary-agent",
        "system_prompt": (
            "You are a synthesis specialist. Summarise the given information into a clear, "
            "executive-level summary. Include: Key takeaway, Supporting evidence, Recommendation."
        ),
        "task_type": "synthesis",
        "fallback": "data-agent",
    },
}


def run_specialist(agent_name: str, task: str, injected_failure: str = "none") -> SpecialistResult:
    """
    Run a specialist agent. May inject failures for testing.
    Returns a SpecialistResult regardless of outcome.
    """
    spec = SPECIALISTS[agent_name]
    start = time.time()

    # Inject timeout failure
    if injected_failure == "timeout":
        elapsed_ms = (time.time() - start) * 1000
        console.print(f"    [red]TIMEOUT[/] {agent_name} — simulated timeout after {elapsed_ms:.0f}ms")
        return SpecialistResult(
            agent_name=agent_name,
            task_type=spec["task_type"],
            output="",
            confidence=0.0,
            processing_time_ms=elapsed_ms,
            status=AgentStatus.FAILED,
            failure_mode=FailureMode.TIMEOUT,
        )

    # Inject low-confidence failure
    if injected_failure == "low_confidence":
        output = llm_call(spec["system_prompt"], task, temperature=0.9)
        elapsed_ms = (time.time() - start) * 1000
        low_conf = round(random.uniform(0.1, 0.35), 2)
        console.print(f"    [yellow]LOW CONFIDENCE[/] {agent_name} — confidence={low_conf}")
        return SpecialistResult(
            agent_name=agent_name,
            task_type=spec["task_type"],
            output=output,
            confidence=low_conf,
            processing_time_ms=elapsed_ms,
            status=AgentStatus.DEGRADED,
            failure_mode=FailureMode.LOW_CONFIDENCE,
        )

    # Normal execution
    try:
        output = llm_call(spec["system_prompt"], task, temperature=0.6)
        elapsed_ms = (time.time() - start) * 1000
        confidence = round(random.uniform(0.75, 0.95), 2)
        console.print(f"    [green]SUCCESS[/] {agent_name} — confidence={confidence}")
        return SpecialistResult(
            agent_name=agent_name,
            task_type=spec["task_type"],
            output=output,
            confidence=confidence,
            processing_time_ms=elapsed_ms,
            status=AgentStatus.SUCCESS,
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        elapsed_ms = (time.time() - start) * 1000
        console.print(f"    [red]EXCEPTION[/] {agent_name} — {str(exc)[:60]}")
        return SpecialistResult(
            agent_name=agent_name,
            task_type=spec["task_type"],
            output="",
            confidence=0.0,
            processing_time_ms=elapsed_ms,
            status=AgentStatus.FAILED,
            failure_mode=FailureMode.EXCEPTION,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  SUPERVISOR WORKFLOW (LangGraph StateGraph)
# ═══════════════════════════════════════════════════════════════════════════

def decompose_task(state: dict) -> dict:
    """Supervisor decomposes the main task into subtasks for each specialist."""
    task = state["task"]
    console.print(f"\n[bold cyan]SUPERVISOR[/] decomposing task: {task[:60]}...")

    subtasks = [
        f"Gather key data and statistics about: {task}",
        f"Analyse trends and patterns related to: {task}",
        f"Synthesise findings into an executive summary about: {task}",
    ]

    decision = DelegationDecision(
        agent_name="supervisor",
        task_type="decomposition",
        action="decompose",
        reasoning=f"Broke task into {len(subtasks)} subtasks for parallel specialist execution",
    )

    return {
        "subtasks": subtasks,
        "delegation_log": state.get("delegation_log", []) + [decision.model_dump()],
    }


def delegate_to_specialists(state: dict) -> dict:
    """Delegate subtasks to specialists, handling failures with re-routing."""
    subtasks = state["subtasks"]
    injected = state.get("injected_failures", {})
    agent_names = list(SPECIALISTS.keys())
    results = []
    failed_agents = []
    delegation_log = list(state.get("delegation_log", []))
    total_retries = 0

    for i, (agent_name, subtask) in enumerate(zip(agent_names, subtasks)):
        console.print(f"\n  [bold]Delegating to {agent_name}[/] (subtask {i+1}/{len(subtasks)})")

        # Get injected failure for this agent
        failure = injected.get(agent_name, "none")

        # Log delegation decision
        decision = DelegationDecision(
            agent_name=agent_name,
            task_type=SPECIALISTS[agent_name]["task_type"],
            action="delegate",
            reasoning=f"Primary specialist for {SPECIALISTS[agent_name]['task_type']}",
        )
        delegation_log.append(decision.model_dump())

        # Run the specialist
        result = run_specialist(agent_name, subtask, failure)

        # Check if we need to re-route
        if result.status == AgentStatus.FAILED or result.confidence < CONFIDENCE_THRESHOLD:
            failure_type = result.failure_mode.value
            console.print(f"  [yellow]SUPERVISOR[/] detected failure: {failure_type} from {agent_name}")

            failed_agents.append(agent_name)
            fallback = SPECIALISTS[agent_name]["fallback"]

            # Log re-routing decision
            reroute_decision = DelegationDecision(
                agent_name="supervisor",
                task_type="re-routing",
                action="re-route",
                reasoning=(
                    f"{agent_name} failed with {failure_type}. "
                    f"Re-routing to fallback agent '{fallback}' because it covers "
                    f"complementary capabilities."
                ),
                failure_detected=failure_type,
                alternative_agent=fallback,
            )
            delegation_log.append(reroute_decision.model_dump())

            # Try fallback agent (no injected failure for fallback)
            console.print(f"  [bold]Re-routing to {fallback}[/] (fallback)")
            fallback_result = run_specialist(fallback, subtask, "none")
            total_retries += 1

            if fallback_result.status == AgentStatus.SUCCESS and fallback_result.confidence >= CONFIDENCE_THRESHOLD:
                console.print(f"  [green]SUPERVISOR[/] fallback {fallback} succeeded")
                fallback_result.attempt_number = 2
                results.append(fallback_result.model_dump())
            else:
                # Both failed — graceful degradation
                console.print(f"  [red]SUPERVISOR[/] all alternatives failed for subtask {i+1}")
                graceful_decision = DelegationDecision(
                    agent_name="supervisor",
                    task_type="graceful_degradation",
                    action="fallback",
                    reasoning=(
                        f"Both {agent_name} and {fallback} failed. "
                        f"Producing degraded output from best available partial result."
                    ),
                    failure_detected=f"double_failure:{agent_name},{fallback}",
                )
                delegation_log.append(graceful_decision.model_dump())

                # Use whatever partial result we have
                best = result if result.confidence > fallback_result.confidence else fallback_result
                best_dict = best.model_dump()
                best_dict["status"] = "degraded"
                best_dict["output"] = best.output or f"[Degraded] Partial results for: {subtask[:50]}"
                results.append(best_dict)
        else:
            results.append(result.model_dump())

    return {
        "specialist_results": results,
        "failed_agents": failed_agents,
        "delegation_log": delegation_log,
        "total_retries": total_retries,
    }


def aggregate_results(state: dict) -> dict:
    """Aggregate all specialist results into a final output."""
    results = state["specialist_results"]
    failed = state.get("failed_agents", [])

    console.print(f"\n[bold cyan]SUPERVISOR[/] aggregating {len(results)} results...")

    # Build sections from results
    sections = []
    for r in results:
        status_icon = "[green]OK[/]" if r["status"] == "success" else "[yellow]DEGRADED[/]"
        sections.append(f"[{r['agent_name']}] ({status_icon}, confidence={r['confidence']})\n{r['output']}")

    # Use LLM to create final synthesis if we have good results
    good_results = [r for r in results if r.get("confidence", 0) >= CONFIDENCE_THRESHOLD]

    if good_results:
        combined_text = "\n\n---\n\n".join([r["output"] for r in good_results if r["output"]])
        final = llm_call(
            "You are a supervisor agent. Synthesise these specialist outputs into one coherent, "
            "well-structured final report. Note any areas where information was limited.",
            f"Specialist outputs:\n{combined_text}",
            temperature=0.4,
        )
    else:
        final = (
            "DEGRADED OUTPUT: All specialist agents experienced failures. "
            "Partial results were collected but confidence is low. "
            f"Failed agents: {', '.join(failed)}. "
            "Recommendation: retry with different parameters or review agent configurations."
        )

    # Determine system status
    if len(failed) == 0:
        status = "success"
    elif len(failed) < len(SPECIALISTS):
        status = "partial_success"
    else:
        status = "degraded"

    return {
        "final_output": final,
        "system_status": status,
    }


def build_supervisor_graph() -> StateGraph:
    """Build the LangGraph StateGraph for the supervisor workflow."""
    graph = StateGraph(dict)
    graph.add_node("decompose", decompose_task)
    graph.add_node("delegate", delegate_to_specialists)
    graph.add_node("aggregate", aggregate_results)

    graph.add_edge(START, "decompose")
    graph.add_edge("decompose", "delegate")
    graph.add_edge("delegate", "aggregate")
    graph.add_edge("aggregate", END)

    return graph.compile()


# ═══════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def render_decision_log(log: list[dict]) -> None:
    """Render the supervisor's decision log as a rich table."""
    table = Table(
        title="Supervisor Decision Log",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold cyan",
    )
    table.add_column("#", style="dim", width=4)
    table.add_column("Agent", style="bold")
    table.add_column("Action", style="cyan")
    table.add_column("Reasoning", style="white", max_width=50)
    table.add_column("Failure", style="red")
    table.add_column("Alt Agent", style="yellow")

    for i, d in enumerate(log, 1):
        table.add_row(
            str(i),
            d.get("agent_name", "N/A"),
            d.get("action", "N/A"),
            d.get("reasoning", "")[:50],
            d.get("failure_detected", "-") or "-",
            d.get("alternative_agent", "-") or "-",
        )
    console.print(table)


def render_results(results: list[dict]) -> None:
    """Render specialist results as panels."""
    for r in results:
        status = r.get("status", "unknown")
        if status == "success":
            border = "green"
            icon = "SUCCESS"
        elif status == "degraded":
            border = "yellow"
            icon = "DEGRADED"
        else:
            border = "red"
            icon = "FAILED"

        output = r.get("output", "No output")
        if len(output) > 300:
            output = output[:300] + "..."

        console.print(Panel(
            output,
            title=f"[bold]{r['agent_name']}[/] [{icon}]",
            subtitle=f"Confidence: {r.get('confidence', 0):.2f} | Time: {r.get('processing_time_ms', 0):.0f}ms | Attempt: {r.get('attempt_number', 1)}",
            border_style=border,
        ))


# ═══════════════════════════════════════════════════════════════════════════
#  CLI COMMANDS
# ═══════════════════════════════════════════════════════════════════════════

def _run_pipeline(task: str, injected_failures: dict[str, str] | None = None) -> dict:
    """Run the supervisor pipeline with optional injected failures."""
    state = {
        "task": task,
        "subtasks": [],
        "delegation_log": [],
        "specialist_results": [],
        "failed_agents": [],
        "current_agent_idx": 0,
        "final_output": "",
        "total_retries": 0,
        "system_status": "running",
        "injected_failures": injected_failures or {},
    }

    # Run the 3-step pipeline directly
    updates = decompose_task(state)
    state.update(updates)

    updates = delegate_to_specialists(state)
    state.update(updates)

    updates = aggregate_results(state)
    state.update(updates)

    return state


@app.command()
def demo():
    """Run the supervisor with all agents operating normally."""
    console.print(Rule("[bold cyan]Lab 5.2 — Supervisor with Failure Recovery (Normal Mode)[/]"))
    console.print()

    # Show architecture
    tree = Tree("[bold]Supervisor Architecture[/]")
    sup = tree.add("[cyan]Supervisor Agent[/]")
    pool = sup.add("[yellow]Specialist Pool[/]")
    pool.add("[green]data-agent[/]     → Data gathering & statistics")
    pool.add("[green]analytics-agent[/] → Trend analysis & patterns")
    pool.add("[green]summary-agent[/]   → Executive synthesis")
    handler = sup.add("[red]Failure Handler[/]")
    handler.add("Timeout detection → re-route to fallback")
    handler.add("Low-confidence detection → re-route to fallback")
    handler.add("Double failure → graceful degradation")
    console.print(Panel(tree, title="System Architecture", border_style="blue"))
    console.print()

    result = _run_pipeline(
        "The current state and future trajectory of generative AI in enterprise software development"
    )

    console.print()
    console.print(Rule("[bold green]Specialist Results[/]"))
    render_results(result["specialist_results"])

    console.print()
    render_decision_log(result["delegation_log"])

    console.print()
    console.print(Panel(
        result["final_output"],
        title=f"[bold]Final Output — Status: {result['system_status'].upper()}[/]",
        subtitle=f"Retries: {result['total_retries']} | Failed Agents: {len(result['failed_agents'])}",
        border_style="green" if result["system_status"] == "success" else "yellow",
    ))


@app.command(name="inject-timeout")
def inject_timeout():
    """Run with timeout failure injected into data-agent."""
    console.print(Rule("[bold red]Lab 5.2 — Injected Failure: TIMEOUT[/]"))
    console.print()
    console.print("[bold yellow]Injecting TIMEOUT failure into data-agent...[/]\n")

    result = _run_pipeline(
        "Analysis of cloud computing market growth and key players in 2025",
        injected_failures={"data-agent": "timeout"},
    )

    console.print()
    console.print(Rule("[bold]Specialist Results[/]"))
    render_results(result["specialist_results"])

    console.print()
    render_decision_log(result["delegation_log"])

    console.print()
    console.print(Panel(
        result["final_output"],
        title=f"[bold]Final Output — Status: {result['system_status'].upper()}[/]",
        subtitle=f"Retries: {result['total_retries']} | Failed: {result['failed_agents']}",
        border_style="yellow",
    ))


@app.command(name="inject-low-confidence")
def inject_low_confidence():
    """Run with low-confidence failure injected into analytics-agent."""
    console.print(Rule("[bold red]Lab 5.2 — Injected Failure: LOW CONFIDENCE[/]"))
    console.print()
    console.print("[bold yellow]Injecting LOW CONFIDENCE into analytics-agent...[/]\n")

    result = _run_pipeline(
        "Impact of artificial intelligence on cybersecurity threat detection",
        injected_failures={"analytics-agent": "low_confidence"},
    )

    console.print()
    console.print(Rule("[bold]Specialist Results[/]"))
    render_results(result["specialist_results"])

    console.print()
    render_decision_log(result["delegation_log"])

    console.print()
    console.print(Panel(
        result["final_output"],
        title=f"[bold]Final Output — Status: {result['system_status'].upper()}[/]",
        subtitle=f"Retries: {result['total_retries']} | Failed: {result['failed_agents']}",
        border_style="yellow",
    ))


@app.command(name="stress-test")
def stress_test():
    """Inject BOTH failure modes simultaneously — the system must not crash."""
    console.print(Rule("[bold red]Lab 5.2 — STRESS TEST: Multiple Simultaneous Failures[/]"))
    console.print()
    console.print("[bold yellow]Injecting TIMEOUT into data-agent AND LOW CONFIDENCE into analytics-agent[/]\n")

    result = _run_pipeline(
        "Emerging trends in sustainable energy technology and market adoption",
        injected_failures={
            "data-agent": "timeout",
            "analytics-agent": "low_confidence",
        },
    )

    console.print()
    console.print(Rule("[bold]Specialist Results[/]"))
    render_results(result["specialist_results"])

    console.print()
    render_decision_log(result["delegation_log"])

    console.print()
    status = result["system_status"]
    border = "green" if status == "success" else "yellow" if status == "partial_success" else "red"
    console.print(Panel(
        result["final_output"],
        title=f"[bold]Final Output — Status: {status.upper()}[/]",
        subtitle=f"Retries: {result['total_retries']} | Failed: {result['failed_agents']}",
        border_style=border,
    ))

    # Stress test summary
    console.print()
    summary_table = Table(title="Stress Test Summary", box=box.ROUNDED, show_lines=True)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value", justify="center")
    summary_table.add_row("System crashed?", "[green]NO[/]")
    summary_table.add_row("Final output produced?", "[green]YES[/]" if result["final_output"] else "[red]NO[/]")
    summary_table.add_row("Agents failed", str(len(result["failed_agents"])))
    summary_table.add_row("Total retries", str(result["total_retries"]))
    summary_table.add_row("Decisions logged", str(len(result["delegation_log"])))
    summary_table.add_row("System status", status.upper())
    console.print(summary_table)
    console.print("\n[bold green]System survived stress test without crashing.[/]\n")


if __name__ == "__main__":
    app()
