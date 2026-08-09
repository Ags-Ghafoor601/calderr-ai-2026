#!/usr/bin/env python3
"""
CalderR Internship – Week 5, Project 5-I-A
=============================================
Autonomous Competitive Intelligence Agent — Main Orchestrator

WHAT THIS PROJECT BUILDS:
-------------------------
A multi-agent system that takes a company name and autonomously
researches it from multiple angles:
  • Market position & competitive landscape
  • Product features & differentiators
  • Technology stack & technical position
  • Recent news & developments
  • Public/analyst sentiment

Then synthesises all findings into a structured intelligence briefing,
detecting and resolving contradictions between agents.

ARCHITECTURE:
    ┌──────────────────────┐
    │  ORCHESTRATOR AGENT  │ ← Plans research, assigns sub-questions
    └──────────┬───────────┘
               │ fan-out (parallel)
    ┌──────────▼──────────────────────────────────────┐
    │              SPECIALIST AGENTS                   │
    │  ┌────────┐ ┌────────┐ ┌──────┐ ┌────┐ ┌─────┐ │
    │  │Market  │ │Product │ │ Tech │ │News│ │Sent.│ │
    │  │Agent   │ │Agent   │ │Agent │ │Agt │ │Agt  │ │
    │  └────┬───┘ └───┬────┘ └──┬───┘ └─┬──┘ └──┬──┘ │
    └───────┼─────────┼────────┼───────┼───────┼────┘
            │         │        │       │       │
            └─────────┴────────┴───────┴───────┘
                              │
               ┌──────────────▼──────────────┐
               │     CONFLICT RESOLVER       │ ← Flags contradictions
               └──────────────┬──────────────┘
                              │
               ┌──────────────▼──────────────┐
               │      SYNTHESIS AGENT        │ ← Final briefing
               └──────────────┬──────────────┘
                              │
                        FINAL REPORT

Run:
    python projects/competitive_intel/main.py demo
    python projects/competitive_intel/main.py analyse "Tesla"
    python projects/competitive_intel/main.py report "OpenAI"
    python projects/competitive_intel/main.py sample-reports
"""

# pylint: disable=line-too-long, too-many-locals, wrong-import-position, duplicate-code, import-error, too-many-branches, too-many-statements, redefined-outer-name, unused-variable, unused-import
import io
import os
import sys
import json
import time
import concurrent.futures
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
ROOT_DIR = PROJECT_DIR.parent.parent
sys.path.insert(0, str(ROOT_DIR))

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

load_dotenv(ROOT_DIR / ".env")

from projects.competitive_intel.models import (
    SynthesisReport, AgentReport,
)
from projects.competitive_intel.agents import (
    OrchestratorAgent, MarketAgent, ProductAgent, TechStackAgent,
    NewsAgent, SentimentAgent, ConflictResolverAgent, SynthesisAgent,
)

console = Console()
app = typer.Typer(help="Project 5-I-A — Autonomous Competitive Intelligence Agent")

SAMPLE_DIR = PROJECT_DIR / "sample_reports"


# ═══════════════════════════════════════════════════════════════════════════
#  PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_intelligence_pipeline(company_name: str, verbose: bool = True) -> SynthesisReport:
    """
    Run the full competitive intelligence pipeline.

    1. Orchestrator plans research
    2. 5 specialist agents run (sequentially to respect API rate limits)
    3. Conflict resolver detects contradictions
    4. Synthesis agent produces final briefing
    """
    total_start = time.time()

    # Agents
    orchestrator = OrchestratorAgent()
    specialists = {
        "market-agent": MarketAgent(),
        "product-agent": ProductAgent(),
        "tech-agent": TechStackAgent(),
        "news-agent": NewsAgent(),
        "sentiment-agent": SentimentAgent(),
    }
    conflict_resolver = ConflictResolverAgent()
    synthesis_agent = SynthesisAgent()

    # Phase 1: Plan research
    if verbose:
        console.print(Rule("[bold cyan]Phase 1: Research Planning[/]"))
    requests = orchestrator.plan_research(company_name)
    if verbose:
        console.print(f"  [green]Orchestrator[/] created {len(requests)} research requests\n")

    # Phase 2: Run specialists
    if verbose:
        console.print(Rule("[bold cyan]Phase 2: Specialist Research (Fan-Out)[/]"))

    reports: list[AgentReport] = []

    def _run_agent(req):
        target_agent = req.context.get("target_agent", "unknown")
        if verbose:
            console.print(f"  Starting [cyan]{target_agent}[/]...")

        try:
            if target_agent == "market-agent":
                result_report = specialists["market-agent"].research(req)
            elif target_agent == "product-agent":
                result_report = specialists["product-agent"].research(req)
            elif target_agent == "tech-agent":
                result_report = specialists["tech-agent"].research(req)
            elif target_agent == "news-agent":
                result_report = specialists["news-agent"].research(req)
            elif target_agent == "sentiment-agent":
                result_report = specialists["sentiment-agent"].research(req)
            else:
                return None

            if verbose:
                console.print(
                    f"    [green]Done[/] [cyan]{target_agent}[/] — confidence: {result_report.confidence:.2f}, "
                    f"time: {result_report.processing_time_ms:.0f}ms"
                )
            return result_report
        except Exception as exc:  # pylint: disable=broad-exception-caught
            if verbose:
                console.print(f"    [red]Error[/] in [cyan]{target_agent}[/]: {str(exc)[:80]}")
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(requests)) as executor:
        futures = [executor.submit(_run_agent, req) for req in requests]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res:
                reports.append(res)

    if verbose:
        console.print(f"\n  [bold]{len(reports)}/{len(requests)} agents completed successfully[/]\n")

    # Phase 3: Conflict detection
    if verbose:
        console.print(Rule("[bold cyan]Phase 3: Conflict Detection[/]"))

    conflicts = conflict_resolver.detect_conflicts(reports)

    if conflicts:
        if verbose:
            console.print(f"  [yellow]{len(conflicts)} conflict(s) detected[/]")
        # Resolve each conflict
        for conf in conflicts:
            resolved_conf = conflict_resolver.resolve_conflict(conf, reports)
            if verbose:
                console.print(f"    [green]Resolved[/]: {resolved_conf.topic[:60]}")
    else:
        if verbose:
            console.print("  [green]No conflicts detected between agent reports[/]")
    if verbose:
        console.print()

    # Phase 4: Synthesis
    if verbose:
        console.print(Rule("[bold cyan]Phase 4: Intelligence Synthesis[/]"))

    synthesis = synthesis_agent.synthesise(company_name, reports, conflicts)

    total_time = (time.time() - total_start) * 1000
    synthesis.total_processing_time_ms = round(total_time, 1)

    if verbose:
        console.print(f"  [green]Synthesis complete[/] — total time: {total_time:.0f}ms\n")

    return synthesis


# ═══════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def render_report(report: SynthesisReport) -> None:
    """Render a full intelligence report with rich formatting."""
    console.print()
    console.print(Rule(f"[bold green]Competitive Intelligence Report: {report.company_name}[/]"))
    console.print()

    # Executive Summary
    console.print(Panel(
        report.executive_summary,
        title="[bold]Executive Summary[/]",
        border_style="green",
    ))

    # Key Insights
    if report.key_insights:
        insights_text = "\n".join([f"  {i}. {insight}" for i, insight in enumerate(report.key_insights, 1)])
        console.print(Panel(insights_text, title="[bold cyan]Key Insights[/]", border_style="cyan"))

    # Detailed Analysis sections
    sections = [
        ("Market Analysis", report.market_analysis, "blue"),
        ("Product Analysis", report.product_analysis, "magenta"),
        ("Technology Analysis", report.technology_analysis, "yellow"),
        ("News Summary", report.news_summary, "cyan"),
        ("Sentiment Analysis", report.sentiment_analysis, "green"),
    ]

    for title, content, color in sections:
        if content and content != "N/A":
            display_content = content if len(content) <= 500 else content[:500] + "..."
            console.print(Panel(display_content, title=f"[bold]{title}[/]", border_style=color))

    # Conflicts
    if report.conflicts_detected:
        conflict_table = Table(title="Detected Conflicts", box=box.ROUNDED, show_lines=True)
        conflict_table.add_column("Topic", style="bold")
        conflict_table.add_column("Agent A", style="cyan")
        conflict_table.add_column("Agent B", style="yellow")
        conflict_table.add_column("Severity")
        conflict_table.add_column("Resolution", max_width=40)

        for c in report.conflicts_detected:
            sev = c.get("severity", "medium")
            sev_color = "red" if sev == "high" else "yellow" if sev == "medium" else "green"
            conflict_table.add_row(
                str(c.get("topic", ""))[:30],
                str(c.get("agent_a", "")),
                str(c.get("agent_b", "")),
                f"[{sev_color}]{sev}[/]",
                str(c.get("resolution", "Unresolved"))[:40],
            )
        console.print(conflict_table)

    # Risk Factors
    if report.risk_factors:
        risks_text = "\n".join([f"  [red]![/] {risk}" for risk in report.risk_factors])
        console.print(Panel(risks_text, title="[bold red]Risk Factors[/]", border_style="red"))

    # Recommendations
    if report.recommendations:
        recs_text = "\n".join([f"  {i}. {rec}" for i, rec in enumerate(report.recommendations, 1)])
        console.print(Panel(recs_text, title="[bold green]Recommendations[/]", border_style="green"))

    # Meta
    console.print(Panel(
        f"[bold]Confidence:[/] {report.overall_confidence:.2f}  |  "
        f"[bold]Agents Used:[/] {report.agents_used}  |  "
        f"[bold]Total Time:[/] {report.total_processing_time_ms:.0f}ms  |  "
        f"[bold]Conflicts:[/] {len(report.conflicts_detected)}",
        title="Report Metadata",
        border_style="dim",
    ))


def render_architecture() -> None:
    """Display the system architecture tree."""
    tree = Tree("[bold]Competitive Intelligence Agent — Architecture[/]")
    orch = tree.add("[cyan]OrchestratorAgent[/] — Plans research strategy")
    fan_out = orch.add("[yellow]Fan-Out: 5 Specialist Agents (parallel)[/]")
    fan_out.add("[green]MarketAgent[/]     — Market position, sizing, competitors")
    fan_out.add("[green]ProductAgent[/]    — Products, features, differentiators")
    fan_out.add("[green]TechStackAgent[/]  — Technology choices, strengths, risks")
    fan_out.add("[green]NewsAgent[/]       — Recent developments, events")
    fan_out.add("[green]SentimentAgent[/]  — Public/analyst sentiment")
    conflict = orch.add("[red]ConflictResolver[/] — Detects contradictions")
    synth = orch.add("[magenta]SynthesisAgent[/] — Final intelligence briefing")
    output = tree.add("[bold green]Output[/]")
    output.add("Executive Summary + Detailed Sections")
    output.add("Key Insights + Risk Factors + Recommendations")
    output.add("Conflict Log + Agent Attribution")
    console.print(Panel(tree, title="System Architecture", border_style="blue"))


# ═══════════════════════════════════════════════════════════════════════════
#  CLI COMMANDS
# ═══════════════════════════════════════════════════════════════════════════

@app.command()
def demo():
    """Run a full demo with a sample company."""
    console.print(Rule("[bold cyan]Project 5-I-A — Competitive Intelligence Agent Demo[/]"))
    console.print()
    render_architecture()
    console.print()

    synth_report = run_intelligence_pipeline("Tesla")
    render_report(synth_report)

    # Save
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SAMPLE_DIR / "demo_tesla.json"
    out_path.write_text(json.dumps(synth_report.model_dump(), indent=2, default=str), encoding="utf-8")
    console.print(f"\n[dim]Report saved to {out_path}[/]\n")


@app.command()
def analyse(company: str):
    """Analyse a specific company."""
    console.print(Rule(f"[bold cyan]Analysing: {company}[/]"))
    console.print()

    synth_report = run_intelligence_pipeline(company)
    render_report(synth_report)


@app.command()
def report(company: str):
    """Generate and save a full intelligence report for a company."""
    console.print(Rule(f"[bold cyan]Generating Report: {company}[/]"))
    console.print()

    synth_report = run_intelligence_pipeline(company)
    render_report(synth_report)

    # Save report
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = company.lower().replace(" ", "_").replace(".", "")
    out_path = SAMPLE_DIR / f"report_{safe_name}.json"
    out_path.write_text(json.dumps(synth_report.model_dump(), indent=2, default=str), encoding="utf-8")
    console.print(f"\n[dim]Report saved to {out_path}[/]\n")


@app.command(name="sample-reports")
def sample_reports():
    """Generate intelligence reports for 3 sample companies."""
    console.print(Rule("[bold cyan]Generating Sample Reports (3 companies)[/]"))
    console.print()
    render_architecture()
    console.print()

    companies = ["Tesla", "OpenAI", "Spotify"]
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    for i, company in enumerate(companies, 1):
        console.print(Rule(f"[bold]Company {i}/{len(companies)}: {company}[/]"))
        console.print()

        synth_report = run_intelligence_pipeline(company)
        render_report(synth_report)

        # Save
        safe_name = company.lower().replace(" ", "_")
        out_path = SAMPLE_DIR / f"report_{safe_name}.json"
        out_path.write_text(json.dumps(synth_report.model_dump(), indent=2, default=str), encoding="utf-8")
        console.print(f"[dim]Saved to {out_path}[/]\n")

    console.print(Rule("[bold green]All 3 reports generated successfully![/]"))


@app.command()
def graph():
    """Display the agent architecture."""
    console.print(Rule("[bold cyan]Agent Architecture[/]"))
    console.print()
    render_architecture()


if __name__ == "__main__":
    app()
