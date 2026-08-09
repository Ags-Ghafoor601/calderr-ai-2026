"""
CalderR Internship – Week 5, Project 5-P-A
=============================================
Autonomous AI Research Lab — Agent Definitions

10 agent types executing across 5 phases:
  Phase 1 (Hypothesis):  HypothesisGenerator
  Phase 2 (Evidence):    LiteratureReviewer, DataAnalyst, MethodologyExpert, DomainSpecialist
  Phase 3 (Critique):    CriticAgent
  Phase 4 (Synthesis):   SynthesisAgent
  Phase 5 (Peer Review): PeerReviewAgent

All agents use typed Pydantic schemas for inputs/outputs.
Dynamic agent assembly means not all agents run for every topic.
"""

import os
import json
import time
import re
from typing import Any

from groq import Groq
from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

from projects.research_lab.models import (
    ResearchDomain, AgentRole, AgentStatus, Severity, Verdict,
    Hypothesis, HypothesisReport,
    EvidenceItem, EvidenceReport,
    Critique, CritiqueReport,
    SynthesisReport,
    ReviewComment, PeerReviewReport,
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = "llama-3.1-8b-instant"


def llm_call(system_prompt: str, user_prompt: str, temperature: float = 0.6) -> str:
    """Make a single LLM call via Groq with 4-attempt retry and exponential backoff."""
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
                max_tokens=1500,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = (attempt + 1) * 12
                time.sleep(wait)
            else:
                raise
    return "Unable to generate response after retries."


def _parse_json(raw: str) -> dict:
    """Parse JSON from LLM output, handling markdown code blocks."""
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


def _safe_list(val: Any) -> list:
    """Ensure a value is a list of strings."""
    if isinstance(val, list):
        return [str(x) for x in val]
    if isinstance(val, str):
        return [x.strip() for x in val.split(",") if x.strip()]
    return []


def _safe_float(val: Any, default: float = 0.5) -> float:
    """Safely convert to float in [0, 1]."""
    try:
        return max(0.0, min(1.0, float(val)))
    except (ValueError, TypeError):
        return default


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1: HYPOTHESIS GENERATION
# ═══════════════════════════════════════════════════════════════════════════

class HypothesisGenerator:
    """
    Phase 1 Agent: Generates testable hypotheses for the research topic.
    Uses the domain-specific system prompt from the assembled team.
    """

    def generate(self, topic: str, domain: ResearchDomain,
                 system_prompt: str) -> HypothesisReport:
        """Generate 3 hypotheses with novelty and relevance scores."""
        start = time.time()

        prompt = (
            f"Research topic: {topic}\n\n"
            "Generate EXACTLY 3 hypotheses. For each, provide:\n"
            "Respond in JSON (no markdown):\n"
            '{"hypotheses": [\n'
            '  {"statement": "<hypothesis>", "rationale": "<why this matters>", '
            '"testability": "<how to test>", "novelty_score": <0.0-1.0>, '
            '"domain_relevance": <0.0-1.0>}\n'
            '], "methodology_suggestion": "<recommended research approach>"}'
        )

        raw = llm_call(system_prompt, prompt, temperature=0.7)
        data = _parse_json(raw)
        elapsed = (time.time() - start) * 1000

        hypotheses = []
        for h in data.get("hypotheses", []):
            try:
                hyp = Hypothesis(
                    statement=str(h.get("statement", "Hypothesis not parsed"))[:300],
                    rationale=str(h.get("rationale", "No rationale"))[:300],
                    testability=str(h.get("testability", ""))[:200],
                    novelty_score=_safe_float(h.get("novelty_score", 0.5)),
                    domain_relevance=_safe_float(h.get("domain_relevance", 0.5)),
                )
                hypotheses.append(hyp)
            except Exception:
                continue

        # Fallback if parsing failed
        if not hypotheses:
            hypotheses = [Hypothesis(
                statement=raw[:200] if raw else "Hypothesis generation produced unstructured output",
                rationale="Generated from unstructured LLM response",
            )]

        return HypothesisReport(
            topic=topic,
            domain=domain,
            hypotheses=hypotheses,
            methodology_suggestion=str(data.get("methodology_suggestion", ""))[:500],
            processing_time_ms=round(elapsed, 1),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 2: EVIDENCE GATHERING (multiple agent types)
# ═══════════════════════════════════════════════════════════════════════════

class LiteratureReviewer:
    """Phase 2 Agent: Reviews existing literature and research."""

    def review(self, topic: str, hypotheses: list[Hypothesis],
               system_prompt: str) -> dict:
        """Review literature relevant to the hypotheses."""
        start = time.time()

        hyp_text = "\n".join([f"H{i+1}: {h.statement}" for i, h in enumerate(hypotheses)])
        prompt = (
            f"Topic: {topic}\n\n"
            f"Hypotheses to investigate:\n{hyp_text}\n\n"
            "Review existing literature and respond in JSON (no markdown):\n"
            '{"summary": "<2-3 paragraph literature review>", '
            '"key_findings": ["finding1", "finding2", "finding3"], '
            '"research_gaps": ["gap1", "gap2"], '
            '"evidence_items": [{"hypothesis_id": "<H1/H2/H3>", "summary": "<evidence>", '
            '"supports_hypothesis": true/false, "strength": <0.0-1.0>, '
            '"source_reference": "<source>"}]}'
        )

        raw = llm_call(system_prompt, prompt, temperature=0.5)
        data = _parse_json(raw)
        elapsed = (time.time() - start) * 1000

        evidence_items = []
        for e in data.get("evidence_items", []):
            try:
                item = EvidenceItem(
                    hypothesis_id=str(e.get("hypothesis_id", hypotheses[0].hypothesis_id if hypotheses else "unknown")),
                    source_type="literature",
                    summary=str(e.get("summary", "No summary"))[:300],
                    supports_hypothesis=bool(e.get("supports_hypothesis", True)),
                    strength=_safe_float(e.get("strength", 0.5)),
                    source_reference=str(e.get("source_reference", ""))[:200],
                )
                evidence_items.append(item)
            except Exception:
                continue

        return {
            "agent": "literature-reviewer",
            "summary": str(data.get("summary", raw[:500])),
            "key_findings": _safe_list(data.get("key_findings", [])),
            "research_gaps": _safe_list(data.get("research_gaps", [])),
            "evidence_items": evidence_items,
            "processing_time_ms": round(elapsed, 1),
        }


class DataAnalyst:
    """Phase 2 Agent: Provides quantitative evidence and analysis."""

    def analyse(self, topic: str, hypotheses: list[Hypothesis],
                system_prompt: str) -> dict:
        """Gather and analyse quantitative data."""
        start = time.time()

        hyp_text = "\n".join([f"H{i+1}: {h.statement}" for i, h in enumerate(hypotheses)])
        prompt = (
            f"Topic: {topic}\n\n"
            f"Hypotheses:\n{hyp_text}\n\n"
            "Provide quantitative evidence. Respond in JSON (no markdown):\n"
            '{"analysis_summary": "<data analysis overview>", '
            '"data_points": ["stat1", "stat2", "stat3"], '
            '"evidence_items": [{"hypothesis_id": "<H1/H2/H3>", "summary": "<quantitative evidence>", '
            '"supports_hypothesis": true/false, "strength": <0.0-1.0>}]}'
        )

        raw = llm_call(system_prompt, prompt, temperature=0.5)
        data = _parse_json(raw)
        elapsed = (time.time() - start) * 1000

        evidence_items = []
        for e in data.get("evidence_items", []):
            try:
                item = EvidenceItem(
                    hypothesis_id=str(e.get("hypothesis_id", "unknown")),
                    source_type="data",
                    summary=str(e.get("summary", "No summary"))[:300],
                    supports_hypothesis=bool(e.get("supports_hypothesis", True)),
                    strength=_safe_float(e.get("strength", 0.5)),
                )
                evidence_items.append(item)
            except Exception:
                continue

        return {
            "agent": "data-analyst",
            "analysis_summary": str(data.get("analysis_summary", raw[:500])),
            "data_points": _safe_list(data.get("data_points", [])),
            "evidence_items": evidence_items,
            "processing_time_ms": round(elapsed, 1),
        }


class MethodologyExpert:
    """Phase 2 Agent: Evaluates and recommends research methodology."""

    def evaluate(self, topic: str, hypotheses: list[Hypothesis],
                 system_prompt: str) -> dict:
        """Evaluate methodology and suggest improvements."""
        start = time.time()

        hyp_text = "\n".join([f"H{i+1}: {h.statement}" for i, h in enumerate(hypotheses)])
        prompt = (
            f"Topic: {topic}\n\n"
            f"Hypotheses:\n{hyp_text}\n\n"
            "Evaluate the research methodology needed. Respond in JSON (no markdown):\n"
            '{"methodology_review": "<assessment of appropriate methods>", '
            '"recommended_methods": ["method1", "method2"], '
            '"potential_biases": ["bias1", "bias2"], '
            '"sample_requirements": "<what data/samples are needed>"}'
        )

        raw = llm_call(system_prompt, prompt, temperature=0.4)
        data = _parse_json(raw)
        elapsed = (time.time() - start) * 1000

        return {
            "agent": "methodology-expert",
            "methodology_review": str(data.get("methodology_review", raw[:500])),
            "recommended_methods": _safe_list(data.get("recommended_methods", [])),
            "potential_biases": _safe_list(data.get("potential_biases", [])),
            "sample_requirements": str(data.get("sample_requirements", ""))[:300],
            "processing_time_ms": round(elapsed, 1),
        }


class DomainSpecialist:
    """Phase 2 Agent: Provides deep domain-specific expertise."""

    def analyse(self, topic: str, hypotheses: list[Hypothesis],
                system_prompt: str) -> dict:
        """Provide domain-specific analysis."""
        start = time.time()

        hyp_text = "\n".join([f"H{i+1}: {h.statement}" for i, h in enumerate(hypotheses)])
        prompt = (
            f"Topic: {topic}\n\n"
            f"Hypotheses:\n{hyp_text}\n\n"
            "Provide deep domain expertise. Respond in JSON (no markdown):\n"
            '{"expert_analysis": "<domain-specific deep analysis>", '
            '"key_insights": ["insight1", "insight2", "insight3"], '
            '"evidence_items": [{"hypothesis_id": "<H1/H2/H3>", "summary": "<expert assessment>", '
            '"supports_hypothesis": true/false, "strength": <0.0-1.0>}]}'
        )

        raw = llm_call(system_prompt, prompt, temperature=0.5)
        data = _parse_json(raw)
        elapsed = (time.time() - start) * 1000

        evidence_items = []
        for e in data.get("evidence_items", []):
            try:
                item = EvidenceItem(
                    hypothesis_id=str(e.get("hypothesis_id", "unknown")),
                    source_type="expert_opinion",
                    summary=str(e.get("summary", "No summary"))[:300],
                    supports_hypothesis=bool(e.get("supports_hypothesis", True)),
                    strength=_safe_float(e.get("strength", 0.5)),
                )
                evidence_items.append(item)
            except Exception:
                continue

        return {
            "agent": "domain-specialist",
            "expert_analysis": str(data.get("expert_analysis", raw[:500])),
            "key_insights": _safe_list(data.get("key_insights", [])),
            "evidence_items": evidence_items,
            "processing_time_ms": round(elapsed, 1),
        }


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 3: CRITICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

class CriticAgent:
    """
    Phase 3 Agent: The adversarial reviewer.
    Challenges hypotheses, evidence quality, and methodology.
    This is the agent that makes the research rigorous.
    """

    def critique(self, topic: str, hypothesis_report: dict,
                 evidence_report: dict, system_prompt: str) -> CritiqueReport:
        """Produce a thorough critical analysis."""
        start = time.time()

        # Build context from previous phases
        hypotheses_text = ""
        for h in hypothesis_report.get("hypotheses", []):
            hypotheses_text += f"- {h.get('statement', 'N/A')}\n"

        evidence_text = ""
        for item in evidence_report.get("evidence_items", []):
            if isinstance(item, dict):
                evidence_text += f"- [{item.get('source_type', 'N/A')}] {item.get('summary', 'N/A')[:100]}\n"
            else:
                evidence_text += f"- {str(item)[:100]}\n"

        prompt = (
            f"Topic: {topic}\n\n"
            f"HYPOTHESES:\n{hypotheses_text}\n"
            f"EVIDENCE:\n{evidence_text}\n"
            f"METHODOLOGY: {evidence_report.get('methodology_notes', 'Not specified')}\n\n"
            "Critically evaluate this research. Respond in JSON (no markdown):\n"
            '{"critiques": [{"target_type": "<hypothesis/evidence/methodology>", '
            '"issue": "<specific problem>", '
            '"severity": "<low/medium/high/critical>", '
            '"suggestion": "<how to address>"}], '
            '"methodology_weaknesses": ["weakness1", "weakness2"], '
            '"bias_warnings": ["bias1", "bias2"], '
            '"overall_rigor_score": <0.0-1.0>}'
        )

        raw = llm_call(system_prompt, prompt, temperature=0.4)
        data = _parse_json(raw)
        elapsed = (time.time() - start) * 1000

        critiques = []
        for c in data.get("critiques", []):
            sev_str = str(c.get("severity", "medium")).lower()
            if sev_str not in [s.value for s in Severity]:
                sev_str = "medium"
            try:
                critique = Critique(
                    target_type=str(c.get("target_type", "general"))[:50],
                    issue=str(c.get("issue", "Issue not specified"))[:300],
                    severity=Severity(sev_str),
                    suggestion=str(c.get("suggestion", ""))[:300],
                )
                critiques.append(critique)
            except Exception:
                continue

        # Fallback
        if not critiques:
            critiques = [Critique(
                target_type="general",
                issue=raw[:200] if raw else "Critique produced unstructured output",
                severity=Severity.MEDIUM,
            )]

        return CritiqueReport(
            topic=topic,
            critiques=critiques,
            methodology_weaknesses=_safe_list(data.get("methodology_weaknesses", [])),
            bias_warnings=_safe_list(data.get("bias_warnings", [])),
            overall_rigor_score=_safe_float(data.get("overall_rigor_score", 0.5)),
            processing_time_ms=round(elapsed, 1),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 4: SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════════

class SynthesisAgent:
    """
    Phase 4 Agent: Merges all findings into a coherent research paper.
    Takes input from Phases 1–3 and produces a structured document.
    """

    def synthesise(self, topic: str, hypothesis_report: dict,
                   evidence_report: dict, critique_report: dict,
                   system_prompt: str) -> SynthesisReport:
        """Produce a full research synthesis."""
        start = time.time()

        # Build comprehensive context
        hyp_text = "\n".join([
            f"H{i+1}: {h.get('statement', 'N/A')}"
            for i, h in enumerate(hypothesis_report.get("hypotheses", []))
        ])

        evidence_text = evidence_report.get("literature_summary", "")
        if evidence_report.get("data_analysis_summary"):
            evidence_text += "\n\nDATA: " + evidence_report["data_analysis_summary"]

        critique_text = "\n".join([
            f"[{c.get('severity', 'medium')}] {c.get('issue', 'N/A')}: {c.get('suggestion', '')}"
            for c in critique_report.get("critiques", [])
        ])

        prompt = (
            f"Topic: {topic}\n\n"
            f"HYPOTHESES:\n{hyp_text}\n\n"
            f"EVIDENCE:\n{evidence_text[:600]}\n\n"
            f"CRITIQUES:\n{critique_text[:400]}\n\n"
            "Synthesise into a complete research paper. Respond in JSON (no markdown):\n"
            '{"abstract": "<100-150 word abstract>", '
            '"introduction": "<research context and motivation>", '
            '"methodology": "<research approach used>", '
            '"findings": "<key findings from evidence>", '
            '"discussion": "<interpretation and implications>", '
            '"conclusion": "<main conclusions>", '
            '"limitations": "<study limitations>", '
            '"future_work": "<recommended future research>", '
            '"key_contributions": ["contribution1", "contribution2", "contribution3"], '
            '"overall_confidence": <0.0-1.0>}'
        )

        raw = llm_call(system_prompt, prompt, temperature=0.4)
        data = _parse_json(raw)
        elapsed = (time.time() - start) * 1000

        return SynthesisReport(
            topic=topic,
            abstract=str(data.get("abstract", raw[:300]))[:500],
            introduction=str(data.get("introduction", ""))[:500],
            methodology=str(data.get("methodology", ""))[:500],
            findings=str(data.get("findings", ""))[:500],
            discussion=str(data.get("discussion", ""))[:500],
            conclusion=str(data.get("conclusion", ""))[:500],
            limitations=str(data.get("limitations", ""))[:300],
            future_work=str(data.get("future_work", ""))[:300],
            key_contributions=_safe_list(data.get("key_contributions", [])),
            overall_confidence=_safe_float(data.get("overall_confidence", 0.5)),
            processing_time_ms=round(elapsed, 1),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 5: PEER REVIEW
# ═══════════════════════════════════════════════════════════════════════════

class PeerReviewAgent:
    """
    Phase 5 Agent: Simulates academic peer review.
    Evaluates the final synthesis for publishability.
    """

    def review(self, topic: str, synthesis_report: dict,
               critique_report: dict, system_prompt: str) -> PeerReviewReport:
        """Produce a peer review assessment."""
        start = time.time()

        # Build paper summary for review
        paper_summary = (
            f"ABSTRACT: {synthesis_report.get('abstract', 'N/A')}\n\n"
            f"METHODOLOGY: {synthesis_report.get('methodology', 'N/A')}\n\n"
            f"FINDINGS: {synthesis_report.get('findings', 'N/A')}\n\n"
            f"DISCUSSION: {synthesis_report.get('discussion', 'N/A')}\n\n"
            f"CONCLUSION: {synthesis_report.get('conclusion', 'N/A')}\n\n"
            f"LIMITATIONS: {synthesis_report.get('limitations', 'N/A')}\n\n"
            f"Confidence: {synthesis_report.get('overall_confidence', 'N/A')}\n"
            f"Rigor score from critic: {critique_report.get('overall_rigor_score', 'N/A')}\n"
        )

        prompt = (
            f"Topic: {topic}\n\n"
            f"PAPER TO REVIEW:\n{paper_summary[:1200]}\n\n"
            "Provide a peer review. Respond in JSON (no markdown):\n"
            '{"verdict": "<accept/minor_revisions/major_revisions/reject>", '
            '"overall_score": <0.0-1.0>, '
            '"comments": [{"section": "<section_name>", "comment": "<specific feedback>", '
            '"severity": "<low/medium/high>", "actionable": true/false}], '
            '"strengths": ["strength1", "strength2"], '
            '"weaknesses": ["weakness1", "weakness2"], '
            '"recommendation": "<2-3 sentence overall recommendation>"}'
        )

        raw = llm_call(system_prompt, prompt, temperature=0.4)
        data = _parse_json(raw)
        elapsed = (time.time() - start) * 1000

        # Parse verdict
        verdict_str = str(data.get("verdict", "minor_revisions")).lower()
        verdict_map = {v.value: v for v in Verdict}
        verdict = verdict_map.get(verdict_str, Verdict.MINOR_REVISIONS)

        # Parse comments
        comments = []
        for c in data.get("comments", []):
            sev_str = str(c.get("severity", "medium")).lower()
            if sev_str not in [s.value for s in Severity]:
                sev_str = "medium"
            try:
                comment = ReviewComment(
                    section=str(c.get("section", "general"))[:50],
                    comment=str(c.get("comment", "No comment"))[:300],
                    severity=Severity(sev_str),
                    actionable=bool(c.get("actionable", True)),
                )
                comments.append(comment)
            except Exception:
                continue

        return PeerReviewReport(
            topic=topic,
            verdict=verdict,
            overall_score=_safe_float(data.get("overall_score", 0.5)),
            comments=comments,
            strengths=_safe_list(data.get("strengths", [])),
            weaknesses=_safe_list(data.get("weaknesses", [])),
            recommendation=str(data.get("recommendation", ""))[:500],
            processing_time_ms=round(elapsed, 1),
        )
