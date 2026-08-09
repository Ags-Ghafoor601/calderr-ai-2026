#!/usr/bin/env python3
"""
CalderR Internship – Week 5, Lab 5.3
======================================
Consensus Engine — Confidence-Weighted Voting with Dissent Tracking

WHAT THIS LAB BUILDS:
---------------------
A 4-agent consensus system:
  • 3 specialist agents (TechExpert, BusinessExpert, UserExpert) each produce
    a structured opinion: answer + confidence score (0–1) + reasoning
  • 1 ConsensusAgent aggregates using confidence-weighted voting
  • If no option clears 60% weighted confidence → triggers second round
    with only the top 2 agents
  • Output includes the final answer with a dissent summary when
    agents disagreed

WHAT THIS TEACHES YOU:
----------------------
  • Consensus with explicit confidence produces trustworthy outputs
  • Dissent tracking surfaces disagreement rather than hiding it
  • Confidence-weighted voting gives more weight to surer agents
  • Second-round deliberation improves outcomes when initial consensus
    is weak

ARCHITECTURE:
    ┌────────────────────────────────────────────────────┐
    │                    QUESTION                        │
    └────────────────────┬───────────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         │               │               │
    ┌────▼─────┐   ┌─────▼─────┐   ┌────▼──────┐
    │  Tech    │   │ Business  │   │   User    │
    │  Expert  │   │  Expert   │   │  Expert   │
    │ (conf+   │   │ (conf+    │   │ (conf+    │
    │  reason) │   │  reason)  │   │  reason)  │
    └────┬─────┘   └─────┬─────┘   └────┬──────┘
         │               │               │
    ┌────▼───────────────▼───────────────▼──────┐
    │          CONSENSUS AGENT                   │
    │  confidence-weighted voting                │
    │  ┌──────────────────────────────────┐      │
    │  │  Weighted score >= 60%?          │      │
    │  │   YES → Final answer             │      │
    │  │   NO  → Second round (top 2)     │      │
    │  └──────────────────────────────────┘      │
    └───────────────────┬───────────────────────┘
                        │
    ┌───────────────────▼───────────────────────┐
    │         FINAL VERDICT                      │
    │  answer + confidence + dissent summary     │
    └───────────────────────────────────────────┘

Run:
    python labs/lab_5_3_consensus_engine.py demo
    python labs/lab_5_3_consensus_engine.py debate "Should companies adopt AI coding assistants?"
    python labs/lab_5_3_consensus_engine.py consensus-report
"""

# pylint: disable=line-too-long, too-many-locals, wrong-import-position, duplicate-code
import io
import os
import sys
import json
import time
import re
from pathlib import Path
from enum import Enum
from typing import Optional

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

load_dotenv(ROOT_DIR / ".env")

console = Console()
app = typer.Typer(help="Lab 5.3 — Consensus Engine with Confidence-Weighted Voting")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = "llama-3.1-8b-instant"

CONSENSUS_THRESHOLD = 0.60  # 60% weighted confidence to pass first round

# ─── LLM Helper ────────────────────────────────────────────────────────────
def llm_call(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
    """Make a single LLM call via Groq."""
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
                time.sleep(wait)
            else:
                raise
    return "Unable to generate response after retries."


# ═══════════════════════════════════════════════════════════════════════════
#  PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════

class Stance(str, Enum):
    """Enumeration of possible stances."""
    STRONGLY_AGREE = "strongly_agree"
    AGREE = "agree"
    NEUTRAL = "neutral"
    DISAGREE = "disagree"
    STRONGLY_DISAGREE = "strongly_disagree"


class ExpertOpinion(BaseModel):
    """Structured opinion from a specialist agent."""
    expert_name: str
    perspective: str  # e.g. "technical", "business", "user_experience"
    stance: Stance
    confidence: float = Field(ge=0.0, le=1.0)
    answer: str = Field(min_length=10)
    reasoning: str = Field(min_length=10)
    key_concern: Optional[str] = None
    round_number: int = 1


class ConsensusResult(BaseModel):
    """Final consensus verdict."""
    question: str
    final_answer: str
    final_stance: Stance
    weighted_confidence: float = Field(ge=0.0, le=1.0)
    consensus_reached: bool
    rounds_needed: int
    dissent_summary: str
    individual_opinions: list[dict]
    voting_breakdown: dict[str, float]


# ═══════════════════════════════════════════════════════════════════════════
#  SPECIALIST AGENTS
# ═══════════════════════════════════════════════════════════════════════════

EXPERTS = {
    "tech-expert": {
        "name": "tech-expert",
        "perspective": "technical",
        "system_prompt": (
            "You are a senior technology expert. Evaluate the question from a TECHNICAL perspective. "
            "Consider: implementation complexity, technical risks, scalability, security implications, "
            "and engineering trade-offs.\n\n"
            "You MUST respond in EXACTLY this JSON format (no markdown, no extra text):\n"
            '{"stance": "<one of: strongly_agree, agree, neutral, disagree, strongly_disagree>", '
            '"confidence": <float 0.0-1.0>, '
            '"answer": "<your position in 2-3 sentences>", '
            '"reasoning": "<technical reasoning in 2-3 sentences>", '
            '"key_concern": "<single biggest technical concern>"}'
        ),
    },
    "business-expert": {
        "name": "business-expert",
        "perspective": "business",
        "system_prompt": (
            "You are a senior business strategist. Evaluate the question from a BUSINESS perspective. "
            "Consider: ROI, market positioning, competitive advantage, cost-benefit analysis, "
            "and organizational impact.\n\n"
            "You MUST respond in EXACTLY this JSON format (no markdown, no extra text):\n"
            '{"stance": "<one of: strongly_agree, agree, neutral, disagree, strongly_disagree>", '
            '"confidence": <float 0.0-1.0>, '
            '"answer": "<your position in 2-3 sentences>", '
            '"reasoning": "<business reasoning in 2-3 sentences>", '
            '"key_concern": "<single biggest business concern>"}'
        ),
    },
    "user-expert": {
        "name": "user-expert",
        "perspective": "user_experience",
        "system_prompt": (
            "You are a UX and end-user advocate. Evaluate the question from the USER EXPERIENCE "
            "perspective. Consider: usability, adoption barriers, learning curve, user satisfaction, "
            "and accessibility.\n\n"
            "You MUST respond in EXACTLY this JSON format (no markdown, no extra text):\n"
            '{"stance": "<one of: strongly_agree, agree, neutral, disagree, strongly_disagree>", '
            '"confidence": <float 0.0-1.0>, '
            '"answer": "<your position in 2-3 sentences>", '
            '"reasoning": "<user experience reasoning in 2-3 sentences>", '
            '"key_concern": "<single biggest UX concern>"}'
        ),
    },
}

STANCE_SCORES = {
    "strongly_agree": 1.0,
    "agree": 0.75,
    "neutral": 0.5,
    "disagree": 0.25,
    "strongly_disagree": 0.0,
}


def get_expert_opinion(expert_key: str, question: str, round_num: int = 1,
                       previous_context: str = "") -> ExpertOpinion:
    """Get a structured opinion from an expert agent."""
    expert = EXPERTS[expert_key]

    prompt = f"Question: {question}"
    if previous_context:
        prompt += f"\n\nContext from previous round:\n{previous_context}"
        prompt += "\n\nPlease refine your position considering the other experts' views."

    console.print(f"  [cyan]{expert['name']}[/] deliberating (round {round_num})...")

    raw = llm_call(expert["system_prompt"], prompt, temperature=0.6)

    # Parse JSON response
    try:
        # Clean up potential markdown wrapping
        cleaned = raw.strip()
        if cleaned.startswith("```"):
            cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
            cleaned = re.sub(r"\s*```$", "", cleaned)
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: extract what we can
        data = {
            "stance": "neutral",
            "confidence": 0.5,
            "answer": raw[:200],
            "reasoning": raw[:200],
            "key_concern": "Unable to parse structured response",
        }

    # Validate stance
    stance_str = data.get("stance", "neutral")
    if stance_str not in [s.value for s in Stance]:
        stance_str = "neutral"

    # Validate confidence
    try:
        conf = float(data.get("confidence", 0.5))
        conf = max(0.0, min(1.0, conf))
    except (ValueError, TypeError):
        conf = 0.5

    opinion = ExpertOpinion(
        expert_name=expert["name"],
        perspective=expert["perspective"],
        stance=Stance(stance_str),
        confidence=conf,
        answer=str(data.get("answer", raw[:200]))[:500],
        reasoning=str(data.get("reasoning", "No reasoning provided"))[:500],
        key_concern=str(data.get("key_concern", "None"))[:200] if data.get("key_concern") else None,
        round_number=round_num,
    )

    stance_color = {
        "strongly_agree": "bold green", "agree": "green",
        "neutral": "yellow",
        "disagree": "red", "strongly_disagree": "bold red",
    }
    color = stance_color.get(opinion.stance.value, "white")
    console.print(f"    [{color}]{opinion.stance.value}[/] (confidence: {opinion.confidence:.2f})")

    return opinion


# ═══════════════════════════════════════════════════════════════════════════
#  CONSENSUS ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def compute_weighted_consensus(opinions: list[ExpertOpinion]) -> dict:
    """
    Compute confidence-weighted consensus from expert opinions.

    Returns: {
        "weighted_score": float (0-1 scale, where 1 = unanimous strong agree),
        "weighted_confidence": float (average weighted confidence),
        "stance_distribution": dict[stance -> weighted_percentage],
        "dominant_stance": Stance,
        "consensus_reached": bool,
    }
    """
    total_weight = sum(o.confidence for o in opinions)
    if total_weight == 0:
        total_weight = 1  # avoid division by zero

    # Weighted stance scores
    weighted_score = sum(
        STANCE_SCORES[o.stance.value] * o.confidence for o in opinions
    ) / total_weight

    # Average weighted confidence
    avg_confidence = total_weight / len(opinions)

    # Stance distribution (confidence-weighted)
    stance_weights: dict[str, float] = {}
    for o in opinions:
        key = o.stance.value
        stance_weights[key] = stance_weights.get(key, 0) + o.confidence

    stance_distribution = {k: v / total_weight for k, v in stance_weights.items()}

    # Find dominant stance
    dominant = max(stance_distribution, key=stance_distribution.get)  # type: ignore
    dominant_weight = stance_distribution[dominant]

    # Consensus reached if dominant stance has >= 60% weighted confidence
    consensus_reached = dominant_weight >= CONSENSUS_THRESHOLD

    return {
        "weighted_score": round(weighted_score, 3),
        "weighted_confidence": round(avg_confidence, 3),
        "stance_distribution": {k: round(v, 3) for k, v in stance_distribution.items()},
        "dominant_stance": dominant,
        "dominant_weight": round(dominant_weight, 3),
        "consensus_reached": consensus_reached,
    }


def generate_dissent_summary(opinions: list[ExpertOpinion], dominant_stance: str) -> str:
    """Generate a summary of dissenting opinions."""
    dissenters = [o for o in opinions if o.stance.value != dominant_stance]

    if not dissenters:
        return "No dissent — all experts agreed."

    lines = ["Dissenting views:"]
    for d in dissenters:
        lines.append(
            f"  - {d.expert_name} ({d.perspective}): {d.stance.value} "
            f"(confidence {d.confidence:.2f}) — {d.key_concern or d.reasoning[:80]}"
        )
    return "\n".join(lines)


def run_consensus(question: str) -> ConsensusResult:
    """
    Run the full consensus engine:
    1. Round 1: All 3 experts give opinions
    2. Check if consensus threshold (60%) is met
    3. If not: Round 2 with top 2 experts (with context from round 1)
    4. Produce final verdict with dissent summary
    """
    console.print(Rule("[bold cyan]Round 1 — Initial Deliberation[/]"))
    console.print()

    # Round 1: All experts
    round1_opinions: list[ExpertOpinion] = []
    for expert_key in EXPERTS:
        opinion = get_expert_opinion(expert_key, question, round_num=1)
        round1_opinions.append(opinion)

    # Compute round 1 consensus
    r1_consensus = compute_weighted_consensus(round1_opinions)

    console.print()
    _render_voting_round(round1_opinions, r1_consensus, round_num=1)

    all_opinions = list(round1_opinions)
    final_consensus = r1_consensus
    rounds_needed = 1

    # Check if we need round 2
    if not r1_consensus["consensus_reached"]:
        console.print()
        console.print(Panel(
            f"[yellow]No consensus reached in Round 1.[/]\n"
            f"Dominant stance '{r1_consensus['dominant_stance']}' has only "
            f"{r1_consensus['dominant_weight']:.0%} weighted support (need {CONSENSUS_THRESHOLD:.0%}).\n"
            f"Triggering Round 2 with top 2 experts for refinement.",
            title="[bold yellow]Escalating to Round 2[/]",
            border_style="yellow",
        ))

        # Select top 2 by confidence
        sorted_opinions = sorted(round1_opinions, key=lambda o: o.confidence, reverse=True)
        top2 = sorted_opinions[:2]
        top2_keys = [o.expert_name for o in top2]

        # Build context from round 1
        context_parts = []
        for o in round1_opinions:
            context_parts.append(
                f"{o.expert_name} ({o.perspective}): stance={o.stance.value}, "
                f"confidence={o.confidence:.2f}, answer={o.answer[:100]}"
            )
        context = "\n".join(context_parts)

        console.print()
        console.print(Rule("[bold cyan]Round 2 — Refined Deliberation (Top 2 Experts)[/]"))
        console.print()

        round2_opinions: list[ExpertOpinion] = []
        for expert_key, expert_info in EXPERTS.items():
            if expert_info["name"] in top2_keys:
                opinion = get_expert_opinion(expert_key, question, round_num=2, previous_context=context)
                round2_opinions.append(opinion)

        final_consensus = compute_weighted_consensus(round2_opinions)
        rounds_needed = 2
        all_opinions.extend(round2_opinions)

        console.print()
        _render_voting_round(round2_opinions, final_consensus, round_num=2)

    # Generate dissent summary
    final_round_opinions = round1_opinions if rounds_needed == 1 else round2_opinions  # type: ignore[possibly-undefined]
    dissent = generate_dissent_summary(final_round_opinions, final_consensus["dominant_stance"])

    # Generate final synthesis
    final_stance = Stance(final_consensus["dominant_stance"])

    # Build final answer using LLM
    opinions_text = "\n".join([
        f"{o.expert_name}: {o.stance.value} ({o.confidence:.2f}) — {o.answer}"
        for o in final_round_opinions
    ])

    final_answer = llm_call(
        "You are a consensus synthesis agent. Based on the expert opinions below, "
        "produce a balanced final answer that acknowledges the majority view and notes "
        "any dissent. Be concise (3-4 sentences).",
        f"Question: {question}\n\nExpert opinions:\n{opinions_text}\n\nDissent: {dissent}",
        temperature=0.4,
    )

    return ConsensusResult(
        question=question,
        final_answer=final_answer,
        final_stance=final_stance,
        weighted_confidence=final_consensus["weighted_confidence"],
        consensus_reached=final_consensus["consensus_reached"],
        rounds_needed=rounds_needed,
        dissent_summary=dissent,
        individual_opinions=[o.model_dump() for o in all_opinions],
        voting_breakdown=final_consensus["stance_distribution"],
    )


# ═══════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _render_voting_round(opinions: list[ExpertOpinion], consensus: dict, round_num: int) -> None:
    """Render a voting round as a rich table."""
    table = Table(
        title=f"Round {round_num} — Expert Opinions",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold cyan",
    )
    table.add_column("Expert", style="bold")
    table.add_column("Perspective", style="dim")
    table.add_column("Stance", justify="center")
    table.add_column("Confidence", justify="center")
    table.add_column("Reasoning", max_width=40)
    table.add_column("Key Concern", max_width=30, style="yellow")

    stance_colors = {
        "strongly_agree": "bold green",
        "agree": "green",
        "neutral": "yellow",
        "disagree": "red",
        "strongly_disagree": "bold red",
    }

    for o in opinions:
        color = stance_colors.get(o.stance.value, "white")
        conf_bar = "█" * int(o.confidence * 10) + "░" * (10 - int(o.confidence * 10))
        table.add_row(
            o.expert_name,
            o.perspective,
            f"[{color}]{o.stance.value}[/]",
            f"[{color}]{conf_bar} {o.confidence:.2f}[/]",
            o.reasoning[:40] + ("..." if len(o.reasoning) > 40 else ""),
            (o.key_concern or "-")[:30],
        )

    console.print(table)

    # Consensus metrics
    dist_parts = [f"{k}: {v:.0%}" for k, v in consensus["stance_distribution"].items()]
    consensus_color = "green" if consensus["consensus_reached"] else "yellow"
    console.print(Panel(
        f"[bold]Dominant stance:[/] [{consensus_color}]{consensus['dominant_stance']}[/] "
        f"({consensus['dominant_weight']:.0%} weight)\n"
        f"[bold]Weighted score:[/] {consensus['weighted_score']:.3f}\n"
        f"[bold]Distribution:[/] {' | '.join(dist_parts)}\n"
        f"[bold]Consensus?[/] [{'green' if consensus['consensus_reached'] else 'red'}]"
        f"{'YES' if consensus['consensus_reached'] else 'NO'}[/] "
        f"(threshold: {CONSENSUS_THRESHOLD:.0%})",
        title=f"Round {round_num} Consensus Metrics",
        border_style=consensus_color,
    ))


# ═══════════════════════════════════════════════════════════════════════════
#  CLI COMMANDS
# ═══════════════════════════════════════════════════════════════════════════

@app.command()
def demo():
    """Run the consensus engine with a default question."""
    console.print(Rule("[bold cyan]Lab 5.3 — Consensus Engine Demo[/]"))
    console.print()

    # Show architecture
    tree = Tree("[bold]Consensus Engine Architecture[/]")
    experts_node = tree.add("[yellow]Expert Panel (3 agents)[/]")
    experts_node.add("[cyan]tech-expert[/]     — Technical feasibility & risks")
    experts_node.add("[cyan]business-expert[/] — Business impact & ROI")
    experts_node.add("[cyan]user-expert[/]     — User experience & adoption")
    consensus_node = tree.add("[green]Consensus Agent[/]")
    consensus_node.add("Confidence-weighted voting")
    consensus_node.add(f"Threshold: {CONSENSUS_THRESHOLD:.0%} weighted support")
    consensus_node.add("Second round: top 2 experts if no consensus")
    output_node = tree.add("[magenta]Output[/]")
    output_node.add("Final verdict + dissent summary")
    console.print(Panel(tree, title="System Architecture", border_style="blue"))
    console.print()

    result = run_consensus(
        "Should mid-size companies invest in building custom AI agents for internal operations, "
        "or should they rely on third-party AI-as-a-service platforms?"
    )

    _render_final_verdict(result)


@app.command()
def debate(topic: str):
    """Run the consensus engine on a custom topic."""
    console.print(Rule("[bold cyan]Lab 5.3 — Consensus Engine: Custom Debate[/]"))
    console.print()

    result = run_consensus(topic)
    _render_final_verdict(result)


@app.command(name="consensus-report")
def consensus_report():
    """Run multiple questions and produce a consolidated report."""
    console.print(Rule("[bold cyan]Lab 5.3 — Multi-Question Consensus Report[/]"))
    console.print()

    questions = [
        "Is microservices architecture the right choice for most startup MVPs?",
        "Should engineering teams adopt AI pair programming tools like GitHub Copilot?",
        "Are large language models reliable enough for customer-facing applications?",
    ]

    results: list[ConsensusResult] = []
    for i, q in enumerate(questions, 1):
        console.print(Rule(f"[bold]Question {i}/{len(questions)}[/]"))
        console.print(f"[bold]{q}[/]\n")
        result = run_consensus(q)
        results.append(result)
        console.print()

    # Summary table
    console.print(Rule("[bold green]Consolidated Report[/]"))
    summary_table = Table(
        title="Multi-Question Consensus Summary",
        box=box.ROUNDED,
        show_lines=True,
        title_style="bold cyan",
    )
    summary_table.add_column("#", style="dim", width=4)
    summary_table.add_column("Question", max_width=40)
    summary_table.add_column("Final Stance", justify="center")
    summary_table.add_column("Confidence", justify="center")
    summary_table.add_column("Consensus?", justify="center")
    summary_table.add_column("Rounds", justify="center")
    summary_table.add_column("Dissent", max_width=30)

    for i, r in enumerate(results, 1):
        consensus_icon = "[green]YES[/]" if r.consensus_reached else "[red]NO[/]"
        stance_color = "green" if "agree" in r.final_stance.value else "red" if "disagree" in r.final_stance.value else "yellow"
        summary_table.add_row(
            str(i),
            r.question[:40] + ("..." if len(r.question) > 40 else ""),
            f"[{stance_color}]{r.final_stance.value}[/]",
            f"{r.weighted_confidence:.2f}",
            consensus_icon,
            str(r.rounds_needed),
            r.dissent_summary[:30] + "..." if len(r.dissent_summary) > 30 else r.dissent_summary,
        )

    console.print(summary_table)

    # Save report
    report_path = ROOT_DIR / "labs" / "lab_5_3_consensus_report.json"
    report_data = [r.model_dump() for r in results]
    report_path.write_text(json.dumps(report_data, indent=2, default=str), encoding="utf-8")
    console.print(f"\n[dim]Report saved to {report_path}[/]\n")


def _render_final_verdict(result: ConsensusResult) -> None:
    """Render the final consensus verdict."""
    console.print()
    console.print(Rule("[bold green]Final Verdict[/]"))

    # Main verdict panel
    border = "green" if result.consensus_reached else "yellow"
    consensus_text = "CONSENSUS REACHED" if result.consensus_reached else "NO CONSENSUS (best available)"

    console.print(Panel(
        f"[bold]{result.final_answer}[/]",
        title=f"[bold]{consensus_text}[/]",
        subtitle=(
            f"Stance: {result.final_stance.value} | "
            f"Confidence: {result.weighted_confidence:.2f} | "
            f"Rounds: {result.rounds_needed}"
        ),
        border_style=border,
    ))

    # Voting breakdown
    breakdown_table = Table(title="Voting Breakdown", box=box.SIMPLE_HEAVY)
    breakdown_table.add_column("Stance", style="bold")
    breakdown_table.add_column("Weight", justify="center")
    breakdown_table.add_column("Bar", style="cyan")
    for stance, weight in sorted(result.voting_breakdown.items(), key=lambda x: x[1], reverse=True):
        progress_bar = "█" * int(weight * 20) + "░" * (20 - int(weight * 20))
        breakdown_table.add_row(stance, f"{weight:.0%}", progress_bar)
    console.print(breakdown_table)

    # Dissent panel
    if "No dissent" not in result.dissent_summary:
        console.print(Panel(
            result.dissent_summary,
            title="[bold yellow]Dissent Summary[/]",
            border_style="yellow",
        ))
    else:
        console.print(Panel(
            "[green]All experts aligned — no dissenting views.[/]",
            title="Dissent Summary",
            border_style="green",
        ))
    console.print()


if __name__ == "__main__":
    app()
