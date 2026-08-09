#!/usr/bin/env python3
"""
CalderR Internship – Week 5, Project 5-P-A
=============================================
Autonomous AI Research Lab — Main Orchestrator

WHAT THIS PROJECT BUILDS:
-------------------------
A multi-agent research system that dynamically assembles a team of
specialist agents based on the research domain, then executes a
5-phase pipeline:

  Phase 1: HYPOTHESIS   → Generate testable hypotheses
  Phase 2: EVIDENCE     → Gather literature, data, and expert opinions
  Phase 3: CRITIQUE     → Adversarial review of hypotheses and evidence
  Phase 4: SYNTHESIS    → Merge into a coherent research paper
  Phase 5: PEER REVIEW  → Simulated academic peer review

KEY FEATURES:
  • Dynamic agent assembly: 3–5 domain specialists selected per topic
  • Domain classifier: LLM-first with keyword fallback
  • Critic agent: adversarial reviewer that challenges every finding
  • Peer review: simulated academic review with verdict + score
  • 6 supported domains: technology, medicine, economics, environment,
    social science, and general

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │                  RESEARCH ORCHESTRATOR                      │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │ Domain Classifier → assembles agent team (3–8 agents) │  │
    │  └───────────────────┬───────────────────────────────────┘  │
    │                      │                                      │
    │  ┌───────────────────▼───────────────────────────────────┐  │
    │  │ Phase 1: HYPOTHESIS GENERATION                        │  │
    │  │  • HypothesisGenerator (domain-specific)              │  │
    │  └───────────────────┬───────────────────────────────────┘  │
    │  ┌───────────────────▼───────────────────────────────────┐  │
    │  │ Phase 2: EVIDENCE GATHERING (fan-out)                 │  │
    │  │  • LiteratureReviewer + DataAnalyst                   │  │
    │  │  • MethodologyExpert + DomainSpecialist (if avail.)   │  │
    │  └───────────────────┬───────────────────────────────────┘  │
    │  ┌───────────────────▼───────────────────────────────────┐  │
    │  │ Phase 3: CRITICAL ANALYSIS                            │  │
    │  │  • CriticAgent (adversarial)                          │  │
    │  └───────────────────┬───────────────────────────────────┘  │
    │  ┌───────────────────▼───────────────────────────────────┐  │
    │  │ Phase 4: SYNTHESIS                                    │  │
    │  │  • SynthesisAgent → full research paper               │  │
    │  └───────────────────┬───────────────────────────────────┘  │
    │  ┌───────────────────▼───────────────────────────────────┐  │
    │  │ Phase 5: PEER REVIEW                                  │  │
    │  │  • PeerReviewAgent → verdict + score                  │  │
    │  └───────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘

Run:
    python projects/research_lab/main.py demo
    python projects/research_lab/main.py research "Impact of AI on healthcare"
    python projects/research_lab/main.py batch
    python projects/research_lab/main.py graph
"""

# pylint: disable=line-too-long, trailing-whitespace, too-many-locals, wrong-import-position, duplicate-code, import-error, too-many-branches, too-many-statements, f-string-without-interpolation, unused-import
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

from projects.research_lab.models import (
    ResearchDomain, AgentRole, ResearchPhase, Verdict,
    HypothesisReport, EvidenceReport, CritiqueReport,
    SynthesisReport, PeerReviewReport, FullResearchReport,
    EvidenceItem,
)
from projects.research_lab.agents import (
    HypothesisGenerator, LiteratureReviewer, DataAnalyst,
    MethodologyExpert, DomainSpecialist, CriticAgent,
    SynthesisAgent as SynthesisAgentImpl, PeerReviewAgent,
)
from projects.research_lab.domain_classifier import DomainClassifier

console = Console()
app = typer.Typer(help="Project 5-P-A — Autonomous AI Research Lab")

REPORTS_DIR = PROJECT_DIR / "reports"


# ═══════════════════════════════════════════════════════════════════════════
#  5-PHASE RESEARCH PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

def run_research_pipeline(topic: str, domain_override: str | None = None,
                          verbose: bool = True) -> FullResearchReport:
    """
    Run the full 5-phase research pipeline.

    1. Classify domain → assemble agent team
    2. Phase 1: Generate hypotheses
    3. Phase 2: Gather evidence (fan-out across available agents)
    4. Phase 3: Critical analysis
    5. Phase 4: Synthesis into research paper
    6. Phase 5: Peer review
    """
    total_start = time.time()
    classifier = DomainClassifier()

    # ── Step 0: Domain Classification & Agent Assembly ──────────────────
    if verbose:
        console.print(Rule("[bold cyan]Phase 0: Domain Classification & Agent Assembly[/]"))

    if domain_override and domain_override in [d.value for d in ResearchDomain]:
        domain = ResearchDomain(domain_override)
        if verbose:
            console.print(f"  [yellow]Domain override:[/] {domain.value}")
    else:
        domain = classifier.classify(topic)

    team = classifier.assemble_team(domain)
    agent_names = [a["name"] for a in team]

    if verbose:
        console.print(f"  [green]Detected domain:[/] {domain.value}")
        console.print(f"  [green]Assembled team:[/] {len(team)} agents")
        for a in team:
            console.print(f"    [dim]•[/] {a['name']} — {a['description'][:50]}")
        console.print()

    # Build lookup for easy access to agent prompts
    agent_prompts = {a["role"]: a["system_prompt"] for a in team}

    # ── Phase 1: Hypothesis Generation ──────────────────────────────────
    if verbose:
        console.print(Rule("[bold cyan]Phase 1: Hypothesis Generation[/]"))

    hyp_agent = HypothesisGenerator()
    hyp_prompt = agent_prompts.get(AgentRole.HYPOTHESIS_GENERATOR, team[0]["system_prompt"])
    hypothesis_report = hyp_agent.generate(topic, domain, hyp_prompt)

    if verbose:
        console.print(f"  [green]Generated {len(hypothesis_report.hypotheses)} hypotheses[/]")
        for i, h in enumerate(hypothesis_report.hypotheses, 1):
            console.print(f"    H{i}: {h.statement[:80]}...")
            console.print(f"        Novelty: {h.novelty_score:.2f} | Relevance: {h.domain_relevance:.2f}")
        console.print(f"  [dim]Time: {hypothesis_report.processing_time_ms:.0f}ms[/]\n")

    # ── Phase 2: Evidence Gathering (fan-out) ───────────────────────────
    if verbose:
        console.print(Rule("[bold cyan]Phase 2: Evidence Gathering (Fan-Out)[/]"))

    all_evidence: list[EvidenceItem] = []
    lit_summary = ""
    data_summary = ""
    method_notes = ""
    agents_used_phase2: list[str] = []
    phase2_time = 0.0

    # Run literature reviewer
    def run_lit():
        lit_agents = [a for a in team if a["role"] == AgentRole.LITERATURE_REVIEWER]
        if not lit_agents:
            return None
        if verbose:
            console.print(f"  Starting [cyan]{lit_agents[0]['name']}[/]...")
        lit_reviewer = LiteratureReviewer()
        lit_result = lit_reviewer.review(topic, hypothesis_report.hypotheses, lit_agents[0]["system_prompt"])
        if verbose:
            console.print(f"    [green]Done[/] [cyan]{lit_agents[0]['name']}[/] — {len(lit_result.get('evidence_items', []))} items")
        return {"type": "lit", "name": lit_agents[0]["name"], "result": lit_result}

    # Run data analyst
    def run_data():
        data_agents = [a for a in team if a["role"] == AgentRole.DATA_ANALYST]
        if not data_agents:
            return None
        if verbose:
            console.print(f"  Starting [cyan]{data_agents[0]['name']}[/]...")
        data_analyst = DataAnalyst()
        data_result = data_analyst.analyse(topic, hypothesis_report.hypotheses, data_agents[0]["system_prompt"])
        if verbose:
            console.print(f"    [green]Done[/] [cyan]{data_agents[0]['name']}[/] — {len(data_result.get('evidence_items', []))} items")
        return {"type": "data", "name": data_agents[0]["name"], "result": data_result}

    # Run methodology expert
    def run_method():
        method_agents = [a for a in team if a["role"] == AgentRole.METHODOLOGY_EXPERT]
        if not method_agents:
            return None
        if verbose:
            console.print(f"  Starting [cyan]{method_agents[0]['name']}[/]...")
        method_expert = MethodologyExpert()
        method_result = method_expert.evaluate(topic, hypothesis_report.hypotheses, method_agents[0]["system_prompt"])
        if verbose:
            console.print(f"    [green]Done[/] [cyan]{method_agents[0]['name']}[/] — review complete")
        return {"type": "method", "name": method_agents[0]["name"], "result": method_result}

    # Run domain specialist
    def run_spec():
        specialist_agents = [a for a in team if a["role"] == AgentRole.DOMAIN_SPECIALIST]
        if not specialist_agents:
            return None
        if verbose:
            console.print(f"  Starting [cyan]{specialist_agents[0]['name']}[/]...")
        specialist = DomainSpecialist()
        spec_result = specialist.analyse(topic, hypothesis_report.hypotheses, specialist_agents[0]["system_prompt"])
        if verbose:
            console.print(f"    [green]Done[/] [cyan]{specialist_agents[0]['name']}[/] — {len(spec_result.get('evidence_items', []))} items")
        return {"type": "spec", "name": specialist_agents[0]["name"], "result": spec_result}

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
        futures = [
            executor.submit(run_lit),
            executor.submit(run_data),
            executor.submit(run_method),
            executor.submit(run_spec)
        ]
        
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if not res:
                continue
                
            r_type = res["type"]
            r_name = res["name"]
            r_data = res["result"]
            
            agents_used_phase2.append(r_name)
            phase2_time += r_data.get("processing_time_ms", 0)
            
            if r_type == "lit":
                all_evidence.extend(r_data.get("evidence_items", []))
                lit_summary = r_data.get("summary", "")
            elif r_type == "data":
                all_evidence.extend(r_data.get("evidence_items", []))
                data_summary = r_data.get("analysis_summary", "")
            elif r_type == "method":
                method_notes = r_data.get("methodology_review", "")
            elif r_type == "spec":
                all_evidence.extend(r_data.get("evidence_items", []))

    evidence_report = EvidenceReport(
        topic=topic,
        evidence_items=all_evidence,
        literature_summary=lit_summary,
        data_analysis_summary=data_summary,
        methodology_notes=method_notes,
        agents_used=agents_used_phase2,
        processing_time_ms=round(phase2_time, 1),
    )

    if verbose:
        console.print(
            f"\n  [bold]{len(all_evidence)} total evidence items from "
            f"{len(agents_used_phase2)} agents[/]"
        )
        console.print(f"  [dim]Time: {phase2_time:.0f}ms[/]\n")

    # ── Phase 3: Critical Analysis ──────────────────────────────────────
    if verbose:
        console.print(Rule("[bold cyan]Phase 3: Critical Analysis (Adversarial Review)[/]"))

    critic_agents = [a for a in team if a["role"] == AgentRole.CRITIC]
    critic_prompt = critic_agents[0]["system_prompt"] if critic_agents else (
        "You are a research critic. Find weaknesses and biases."
    )

    critic = CriticAgent()
    critique_report = critic.critique(
        topic,
        hypothesis_report.model_dump(),
        evidence_report.model_dump(),
        critic_prompt,
    )

    if verbose:
        console.print(f"  [yellow]Found {len(critique_report.critiques)} issues[/]")
        for c in critique_report.critiques:
            sev_color = {"low": "green", "medium": "yellow", "high": "red", "critical": "bold red"}.get(c.severity.value, "white")
            console.print(f"    [{sev_color}][{c.severity.value.upper()}][/] {c.issue[:70]}...")
        console.print(f"  [bold]Rigor score:[/] {critique_report.overall_rigor_score:.2f}")
        console.print(f"  [dim]Time: {critique_report.processing_time_ms:.0f}ms[/]\n")

    # ── Phase 4: Synthesis ──────────────────────────────────────────────
    if verbose:
        console.print(Rule("[bold cyan]Phase 4: Research Synthesis[/]"))

    synth_agents = [a for a in team if a["role"] == AgentRole.SYNTHESISER]
    synth_prompt = synth_agents[0]["system_prompt"] if synth_agents else (
        "You are a research synthesiser. Produce a coherent research paper."
    )

    synth_agent = SynthesisAgentImpl()
    synthesis_report = synth_agent.synthesise(
        topic,
        hypothesis_report.model_dump(),
        evidence_report.model_dump(),
        critique_report.model_dump(),
        synth_prompt,
    )

    if verbose:
        console.print(f"  [green]Synthesis complete[/]")
        console.print(f"  [bold]Confidence:[/] {synthesis_report.overall_confidence:.2f}")
        console.print(f"  [bold]Contributions:[/] {len(synthesis_report.key_contributions)}")
        console.print(f"  [dim]Time: {synthesis_report.processing_time_ms:.0f}ms[/]\n")

    # ── Phase 5: Peer Review ────────────────────────────────────────────
    if verbose:
        console.print(Rule("[bold cyan]Phase 5: Peer Review[/]"))

    review_agents = [a for a in team if a["role"] == AgentRole.PEER_REVIEWER]
    review_prompt = review_agents[0]["system_prompt"] if review_agents else (
        "You are an academic peer reviewer."
    )

    reviewer = PeerReviewAgent()
    peer_review = reviewer.review(
        topic,
        synthesis_report.model_dump(),
        critique_report.model_dump(),
        review_prompt,
    )

    if verbose:
        verdict_colors = {
            "accept": "bold green", "minor_revisions": "green",
            "major_revisions": "yellow", "reject": "red",
        }
        v_color = verdict_colors.get(peer_review.verdict.value, "white")
        console.print(f"  [bold]Verdict:[/] [{v_color}]{peer_review.verdict.value.upper()}[/]")
        console.print(f"  [bold]Score:[/] {peer_review.overall_score:.2f}")
        console.print(f"  [bold]Comments:[/] {len(peer_review.comments)}")
        console.print(f"  [dim]Time: {peer_review.processing_time_ms:.0f}ms[/]\n")

    # ── Aggregate Full Report ───────────────────────────────────────────
    total_time = (time.time() - total_start) * 1000

    # Quality score = weighted average of rigor + synthesis confidence + peer score
    quality_score = (
        critique_report.overall_rigor_score * 0.3
        + synthesis_report.overall_confidence * 0.3
        + peer_review.overall_score * 0.4
    )

    full_report = FullResearchReport(
        topic=topic,
        domain=domain,
        hypothesis_report=hypothesis_report.model_dump(),
        evidence_report=evidence_report.model_dump(),
        critique_report=critique_report.model_dump(),
        synthesis_report=synthesis_report.model_dump(),
        peer_review_report=peer_review.model_dump(),
        agents_assembled=agent_names,
        total_agents_used=len(agents_used_phase2) + 3,  # + hypothesis + critic + synth + reviewer
        total_processing_time_ms=round(total_time, 1),
        phases_completed=5,
        overall_quality_score=round(quality_score, 3),
        status="complete",
    )

    return full_report


# ═══════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def render_report(report: FullResearchReport) -> None:
    """Render a full research report with rich formatting."""
    console.print()
    console.print(Rule(f"[bold green]Research Report: {report.topic}[/]"))
    console.print()

    # Meta panel
    console.print(Panel(
        f"[bold]Domain:[/] {report.domain.value}  |  "
        f"[bold]Agents:[/] {report.total_agents_used}  |  "
        f"[bold]Phases:[/] {report.phases_completed}/5  |  "
        f"[bold]Quality:[/] {report.overall_quality_score:.2f}  |  "
        f"[bold]Time:[/] {report.total_processing_time_ms:.0f}ms",
        title="Report Metadata", border_style="dim",
    ))

    # Synthesis sections
    synth = report.synthesis_report
    if synth:
        console.print(Panel(
            str(synth.get("abstract", "N/A")),
            title="[bold]Abstract[/]", border_style="green",
        ))

        sections = [
            ("Introduction", synth.get("introduction", ""), "blue"),
            ("Methodology", synth.get("methodology", ""), "cyan"),
            ("Findings", synth.get("findings", ""), "green"),
            ("Discussion", synth.get("discussion", ""), "yellow"),
            ("Conclusion", synth.get("conclusion", ""), "magenta"),
            ("Limitations", synth.get("limitations", ""), "red"),
            ("Future Work", synth.get("future_work", ""), "dim"),
        ]

        for title, content, color in sections:
            if content:
                display = content if len(content) <= 500 else content[:500] + "..."
                console.print(Panel(display, title=f"[bold]{title}[/]", border_style=color))

        # Key contributions
        contributions = synth.get("key_contributions", [])
        if contributions:
            contrib_text = "\n".join([f"  {i}. {c}" for i, c in enumerate(contributions, 1)])
            console.print(Panel(contrib_text, title="[bold cyan]Key Contributions[/]", border_style="cyan"))

    # Peer Review verdict
    review = report.peer_review_report
    if review:
        verdict = review.get("verdict", "unknown")
        score = review.get("overall_score", 0)
        verdict_color = {
            "accept": "green", "minor_revisions": "green",
            "major_revisions": "yellow", "reject": "red",
        }.get(verdict, "white")

        console.print(Panel(
            f"[bold]Verdict:[/] [{verdict_color}]{verdict.upper()}[/]\n"
            f"[bold]Score:[/] {score:.2f}\n\n"
            f"[bold]Recommendation:[/] {review.get('recommendation', 'N/A')}",
            title="[bold]Peer Review[/]", border_style=verdict_color,
        ))

        # Strengths & Weaknesses
        if review.get("strengths") or review.get("weaknesses"):
            sw_table = Table(box=box.ROUNDED, show_lines=True)
            sw_table.add_column("Strengths", style="green", max_width=40)
            sw_table.add_column("Weaknesses", style="red", max_width=40)

            strengths = review.get("strengths", [])
            weaknesses = review.get("weaknesses", [])
            max_len = max(len(strengths), len(weaknesses))
            for i in range(max_len):
                s = strengths[i] if i < len(strengths) else ""
                w = weaknesses[i] if i < len(weaknesses) else ""
                sw_table.add_row(s, w)
            console.print(sw_table)

    # Assembled agents
    if report.agents_assembled:
        agents_text = ", ".join(report.agents_assembled)
        console.print(Panel(
            agents_text,
            title=f"[bold]Assembled Agent Team ({len(report.agents_assembled)} agents)[/]",
            border_style="dim",
        ))


def render_architecture() -> None:
    """Display the system architecture tree."""
    tree = Tree("[bold]Autonomous AI Research Lab — Architecture[/]")

    orch = tree.add("[cyan]Research Orchestrator[/]")

    # Domain classifier
    dc = orch.add("[yellow]Domain Classifier[/] — LLM + keyword fallback")
    dc.add("[dim]technology | medicine | economics | environment | social_science | general[/]")
    dc.add("[dim]Assembles 3–5 domain specialists + 3 universal agents[/]")

    # Phase 1
    p1 = orch.add("[green]Phase 1: Hypothesis Generation[/]")
    p1.add("[dim]HypothesisGenerator — domain-specific prompts, 3 testable hypotheses[/]")

    # Phase 2
    p2 = orch.add("[green]Phase 2: Evidence Gathering (Fan-Out)[/]")
    p2.add("[dim]LiteratureReviewer — reviews papers, trials, reports[/]")
    p2.add("[dim]DataAnalyst — quantitative evidence, statistics[/]")
    p2.add("[dim]MethodologyExpert — evaluates study design (domain-dependent)[/]")
    p2.add("[dim]DomainSpecialist — deep domain expertise (domain-dependent)[/]")

    # Phase 3
    p3 = orch.add("[red]Phase 3: Critical Analysis (Adversarial)[/]")
    p3.add("[dim]CriticAgent — challenges hypotheses, evidence quality, biases[/]")

    # Phase 4
    p4 = orch.add("[magenta]Phase 4: Synthesis[/]")
    p4.add("[dim]SynthesisAgent — merges all into research paper[/]")
    p4.add("[dim]Abstract | Introduction | Methodology | Findings | Discussion | Conclusion[/]")

    # Phase 5
    p5 = orch.add("[blue]Phase 5: Peer Review[/]")
    p5.add("[dim]PeerReviewAgent — verdict (accept/minor/major/reject) + score[/]")

    output = tree.add("[bold green]Output[/]")
    output.add("Full research paper with peer review verdict")
    output.add("Quality score (rigor × confidence × review)")
    output.add("Agent attribution + processing metrics")

    console.print(Panel(tree, title="System Architecture", border_style="blue"))


# ═══════════════════════════════════════════════════════════════════════════
#  CLI COMMANDS
# ═══════════════════════════════════════════════════════════════════════════

@app.command()
def demo():
    """Run a demo research on a sample topic."""
    console.print(Rule("[bold cyan]Project 5-P-A — Autonomous AI Research Lab Demo[/]"))
    console.print()
    render_architecture()
    console.print()

    report = run_research_pipeline(
        "The impact of large language models on scientific research methodology"
    )
    render_report(report)

    # Save
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "demo_llm_research.json"
    out_path.write_text(json.dumps(report.model_dump(), indent=2, default=str), encoding="utf-8")
    console.print(f"\n[dim]Report saved to {out_path}[/]\n")


@app.command()
def research(topic: str, domain: str = ""):
    """Run the full research pipeline on a custom topic."""
    console.print(Rule(f"[bold cyan]Researching: {topic}[/]"))
    console.print()

    domain_arg = domain if domain else None
    report = run_research_pipeline(topic, domain_override=domain_arg)
    render_report(report)

    # Save
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    safe_name = topic[:30].lower().replace(" ", "_").replace("/", "_")
    out_path = REPORTS_DIR / f"report_{safe_name}.json"
    out_path.write_text(json.dumps(report.model_dump(), indent=2, default=str), encoding="utf-8")
    console.print(f"\n[dim]Report saved to {out_path}[/]\n")


@app.command()
def batch():
    """Generate research reports across 5 different domains."""
    console.print(Rule("[bold cyan]Batch Research — 5 Domains[/]"))
    console.print()
    render_architecture()
    console.print()

    topics = [
        ("The role of transformer architectures in advancing code generation tools", "technology"),
        ("Effectiveness of mRNA vaccine platforms for emerging infectious diseases", "medicine"),
        ("Impact of central bank digital currencies on monetary policy transmission", "economics"),
        ("Carbon capture and storage technologies: scalability and economic viability", "environment"),
        ("Effects of remote work on employee well-being and organizational culture", "social_science"),
    ]

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    for i, (topic, expected_domain) in enumerate(topics, 1):
        console.print(Rule(f"[bold]Research {i}/{len(topics)}: {expected_domain.upper()}[/]"))
        console.print(f"[bold]Topic:[/] {topic}\n")

        report = run_research_pipeline(topic)
        render_report(report)

        # Save
        safe_name = topic[:30].lower().replace(" ", "_").replace("/", "_")
        out_path = REPORTS_DIR / f"report_{safe_name}.json"
        out_path.write_text(json.dumps(report.model_dump(), indent=2, default=str), encoding="utf-8")
        console.print(f"[dim]Saved to {out_path}[/]\n")

    console.print(Rule("[bold green]All 5 reports generated successfully![/]"))


@app.command()
def graph():
    """Display the system architecture."""
    console.print(Rule("[bold cyan]Research Lab Architecture[/]"))
    console.print()
    render_architecture()


if __name__ == "__main__":
    app()
