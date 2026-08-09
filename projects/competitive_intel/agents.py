"""
CalderR Internship – Week 5, Project 5-I-A
=============================================
Competitive Intelligence Agent — Agent Definitions

8 agents:
  • OrchestratorAgent — plans research strategy, delegates
  • MarketAgent — market position, sizing, growth
  • ProductAgent — products, features, differentiators
  • TechStackAgent — technology choices, strengths, risks
  • NewsAgent — recent developments, notable events
  • SentimentAgent — public/analyst sentiment
  • SynthesisAgent — merges findings with confidence scores
  • ConflictResolver — detects and resolves contradictions
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

from projects.competitive_intel.models import (
    ResearchRequest, AgentReport, MarketReport, ProductReport,
    TechStackReport, NewsReport, SentimentReport, SynthesisReport,
    Conflict, ConflictSeverity, Sentiment, AgentStatus, Priority,
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = "llama-3.1-8b-instant"


def llm_call(system_prompt: str, user_prompt: str, temperature: float = 0.6) -> str:
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
                max_tokens=1500,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                import time as _time
                wait = (attempt + 1) * 12
                _time.sleep(wait)
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


# ═══════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR AGENT
# ═══════════════════════════════════════════════════════════════════════════

class OrchestratorAgent:
    """Plans research strategy and creates research requests for each specialist."""

    def plan_research(self, company_name: str) -> list[ResearchRequest]:
        """Create research requests for all specialist agents."""
        research_areas = [
            ("market-agent", f"Analyse the market position, market size, growth trajectory, and competitive landscape for {company_name}."),
            ("product-agent", f"Map the core products, key features, differentiators, and weaknesses of {company_name}."),
            ("tech-agent", f"Infer the technology stack, technical strengths, and technical risks of {company_name}."),
            ("news-agent", f"Find the most recent developments, notable events, and news about {company_name}."),
            ("sentiment-agent", f"Assess public and analyst sentiment towards {company_name}, including positive drivers and risk signals."),
        ]

        requests = []
        for agent_id, focus in research_areas:
            req = ResearchRequest(
                company_name=company_name,
                research_focus=focus,
                priority=Priority.HIGH,
                context={"target_agent": agent_id},
            )
            requests.append(req)

        return requests


# ═══════════════════════════════════════════════════════════════════════════
#  SPECIALIST AGENTS
# ═══════════════════════════════════════════════════════════════════════════

class MarketAgent:
    """Analyses market position and sizing."""

    def research(self, request: ResearchRequest) -> MarketReport:
        start = time.time()
        prompt = (
            f"You are a market analyst. Research {request.company_name} and respond "
            f"in EXACTLY this JSON format (no markdown):\n"
            f'{{"findings": "<2-3 paragraph analysis>", '
            f'"market_size_estimate": "<estimated market size>", '
            f'"market_position": "<leader/challenger/niche/follower>", '
            f'"growth_trajectory": "<growing/stable/declining + explanation>", '
            f'"key_competitors": ["comp1", "comp2", "comp3"], '
            f'"key_data_points": ["point1", "point2", "point3"], '
            f'"confidence": <0.0-1.0>}}'
        )

        raw = llm_call(prompt, request.research_focus)
        data = _parse_json(raw)
        elapsed = (time.time() - start) * 1000

        return MarketReport(
            agent_name="market-agent",
            request_id=request.request_id,
            company_name=request.company_name,
            findings=str(data.get("findings", raw[:500])),
            confidence=min(1.0, max(0.0, float(data.get("confidence", 0.7)))),
            key_data_points=_safe_list(data.get("key_data_points", [])),
            processing_time_ms=round(elapsed, 1),
            market_size_estimate=str(data.get("market_size_estimate", "Unknown")),
            market_position=str(data.get("market_position", "Unknown")),
            growth_trajectory=str(data.get("growth_trajectory", "Unknown")),
            key_competitors=_safe_list(data.get("key_competitors", [])),
        )


class ProductAgent:
    """Maps products, features, and differentiators."""

    def research(self, request: ResearchRequest) -> ProductReport:
        start = time.time()
        prompt = (
            f"You are a product analyst. Research {request.company_name}'s products and respond "
            f"in EXACTLY this JSON format (no markdown):\n"
            f'{{"findings": "<2-3 paragraph analysis>", '
            f'"core_products": ["product1", "product2"], '
            f'"differentiators": ["diff1", "diff2"], '
            f'"weaknesses": ["weakness1", "weakness2"], '
            f'"key_data_points": ["point1", "point2"], '
            f'"confidence": <0.0-1.0>}}'
        )

        raw = llm_call(prompt, request.research_focus)
        data = _parse_json(raw)
        elapsed = (time.time() - start) * 1000

        return ProductReport(
            agent_name="product-agent",
            request_id=request.request_id,
            company_name=request.company_name,
            findings=str(data.get("findings", raw[:500])),
            confidence=min(1.0, max(0.0, float(data.get("confidence", 0.7)))),
            key_data_points=_safe_list(data.get("key_data_points", [])),
            processing_time_ms=round(elapsed, 1),
            core_products=_safe_list(data.get("core_products", [])),
            differentiators=_safe_list(data.get("differentiators", [])),
            weaknesses=_safe_list(data.get("weaknesses", [])),
        )


class TechStackAgent:
    """Infers technology choices and technical position."""

    def research(self, request: ResearchRequest) -> TechStackReport:
        start = time.time()
        prompt = (
            f"You are a technology analyst. Research {request.company_name}'s technology and respond "
            f"in EXACTLY this JSON format (no markdown):\n"
            f'{{"findings": "<2-3 paragraph analysis>", '
            f'"inferred_technologies": ["tech1", "tech2", "tech3"], '
            f'"tech_strengths": ["strength1", "strength2"], '
            f'"tech_risks": ["risk1", "risk2"], '
            f'"key_data_points": ["point1", "point2"], '
            f'"confidence": <0.0-1.0>}}'
        )

        raw = llm_call(prompt, request.research_focus)
        data = _parse_json(raw)
        elapsed = (time.time() - start) * 1000

        return TechStackReport(
            agent_name="tech-agent",
            request_id=request.request_id,
            company_name=request.company_name,
            findings=str(data.get("findings", raw[:500])),
            confidence=min(1.0, max(0.0, float(data.get("confidence", 0.7)))),
            key_data_points=_safe_list(data.get("key_data_points", [])),
            processing_time_ms=round(elapsed, 1),
            inferred_technologies=_safe_list(data.get("inferred_technologies", [])),
            tech_strengths=_safe_list(data.get("tech_strengths", [])),
            tech_risks=_safe_list(data.get("tech_risks", [])),
        )


class NewsAgent:
    """Surfaces recent developments and notable events."""

    def research(self, request: ResearchRequest) -> NewsReport:
        start = time.time()
        prompt = (
            f"You are a news analyst. Research recent news about {request.company_name} and respond "
            f"in EXACTLY this JSON format (no markdown):\n"
            f'{{"findings": "<2-3 paragraph summary of recent news>", '
            f'"recent_developments": ["dev1", "dev2", "dev3"], '
            f'"notable_events": ["event1", "event2"], '
            f'"news_sentiment": "<very_positive/positive/neutral/negative/very_negative>", '
            f'"key_data_points": ["point1", "point2"], '
            f'"confidence": <0.0-1.0>}}'
        )

        raw = llm_call(prompt, request.research_focus)
        data = _parse_json(raw)
        elapsed = (time.time() - start) * 1000

        sentiment_str = str(data.get("news_sentiment", "neutral"))
        if sentiment_str not in [s.value for s in Sentiment]:
            sentiment_str = "neutral"

        return NewsReport(
            agent_name="news-agent",
            request_id=request.request_id,
            company_name=request.company_name,
            findings=str(data.get("findings", raw[:500])),
            confidence=min(1.0, max(0.0, float(data.get("confidence", 0.7)))),
            key_data_points=_safe_list(data.get("key_data_points", [])),
            processing_time_ms=round(elapsed, 1),
            recent_developments=_safe_list(data.get("recent_developments", [])),
            notable_events=_safe_list(data.get("notable_events", [])),
            news_sentiment=Sentiment(sentiment_str),
        )


class SentimentAgent:
    """Gauges public and analyst sentiment."""

    def research(self, request: ResearchRequest) -> SentimentReport:
        start = time.time()
        prompt = (
            f"You are a sentiment analyst. Assess sentiment towards {request.company_name} and respond "
            f"in EXACTLY this JSON format (no markdown):\n"
            f'{{"findings": "<2-3 paragraph sentiment analysis>", '
            f'"overall_sentiment": "<very_positive/positive/neutral/negative/very_negative>", '
            f'"sentiment_drivers": ["driver1", "driver2", "driver3"], '
            f'"risk_signals": ["risk1", "risk2"], '
            f'"key_data_points": ["point1", "point2"], '
            f'"confidence": <0.0-1.0>}}'
        )

        raw = llm_call(prompt, request.research_focus)
        data = _parse_json(raw)
        elapsed = (time.time() - start) * 1000

        sentiment_str = str(data.get("overall_sentiment", "neutral"))
        if sentiment_str not in [s.value for s in Sentiment]:
            sentiment_str = "neutral"

        return SentimentReport(
            agent_name="sentiment-agent",
            request_id=request.request_id,
            company_name=request.company_name,
            findings=str(data.get("findings", raw[:500])),
            confidence=min(1.0, max(0.0, float(data.get("confidence", 0.7)))),
            key_data_points=_safe_list(data.get("key_data_points", [])),
            processing_time_ms=round(elapsed, 1),
            overall_sentiment=Sentiment(sentiment_str),
            sentiment_drivers=_safe_list(data.get("sentiment_drivers", [])),
            risk_signals=_safe_list(data.get("risk_signals", [])),
        )


# ═══════════════════════════════════════════════════════════════════════════
#  CONFLICT RESOLVER AGENT
# ═══════════════════════════════════════════════════════════════════════════

class ConflictResolverAgent:
    """Detects and resolves contradictions between agent reports."""

    def detect_conflicts(self, reports: list[AgentReport]) -> list[Conflict]:
        """Use LLM to detect contradictions between reports."""
        if len(reports) < 2:
            return []

        # Build summary of all reports
        summaries = []
        for r in reports:
            summaries.append(f"{r.agent_name}: {r.findings[:200]}")

        prompt = (
            "You are a conflict detection specialist. Compare these agent reports and "
            "identify ANY contradictions or inconsistencies between them.\n\n"
            "Respond in EXACTLY this JSON format (no markdown):\n"
            '{"conflicts": [{"agent_a": "<name>", "agent_b": "<name>", '
            '"topic": "<what they disagree on>", '
            '"claim_a": "<what agent_a says>", '
            '"claim_b": "<what agent_b says>", '
            '"severity": "<low/medium/high>"}]}\n\n'
            'If no conflicts found, return: {"conflicts": []}'
        )

        raw = llm_call(prompt, "\n\n".join(summaries), temperature=0.3)
        data = _parse_json(raw)

        conflicts = []
        for c in data.get("conflicts", []):
            severity_str = str(c.get("severity", "medium"))
            if severity_str not in [s.value for s in ConflictSeverity]:
                severity_str = "medium"

            conflicts.append(Conflict(
                agent_a=str(c.get("agent_a", "unknown")),
                agent_b=str(c.get("agent_b", "unknown")),
                topic=str(c.get("topic", "Unknown topic")),
                claim_a=str(c.get("claim_a", ""))[:200],
                claim_b=str(c.get("claim_b", ""))[:200],
                severity=ConflictSeverity(severity_str),
            ))

        return conflicts

    def resolve_conflict(self, conflict: Conflict, reports: list[AgentReport]) -> Conflict:
        """Resolve a specific conflict using LLM arbitration."""
        prompt = (
            f"You are an arbitration specialist. Two research agents have contradictory findings.\n\n"
            f"Agent A ({conflict.agent_a}): {conflict.claim_a}\n"
            f"Agent B ({conflict.agent_b}): {conflict.claim_b}\n"
            f"Topic: {conflict.topic}\n\n"
            f"Determine which claim is more likely correct or how to reconcile them. "
            f"Respond with a concise resolution (2-3 sentences)."
        )

        resolution = llm_call(prompt, "Resolve this conflict.", temperature=0.3)
        conflict.resolution = resolution
        conflict.resolved = True
        return conflict


# ═══════════════════════════════════════════════════════════════════════════
#  SYNTHESIS AGENT
# ═══════════════════════════════════════════════════════════════════════════

class SynthesisAgent:
    """Merges all specialist reports into a final intelligence briefing."""

    def synthesise(self, company_name: str, reports: list[AgentReport],
                   conflicts: list[Conflict]) -> SynthesisReport:
        """Produce the final synthesised intelligence report."""
        start = time.time()

        # Build comprehensive context
        report_sections = {}
        for r in reports:
            report_sections[r.agent_type] = r.findings

        conflict_text = ""
        if conflicts:
            conflict_items = []
            for c in conflicts:
                res_text = f" Resolution: {c.resolution}" if c.resolved else ""
                conflict_items.append(f"- {c.topic}: {c.claim_a} vs {c.claim_b}{res_text}")
            conflict_text = "\n".join(conflict_items)

        # Generate executive summary
        prompt = (
            f"You are a senior intelligence analyst. Synthesise these findings about {company_name} "
            f"into a professional executive briefing.\n\n"
            f"Market Analysis:\n{report_sections.get('market', 'N/A')}\n\n"
            f"Product Analysis:\n{report_sections.get('product', 'N/A')}\n\n"
            f"Technology Analysis:\n{report_sections.get('tech_stack', 'N/A')}\n\n"
            f"News Summary:\n{report_sections.get('news', 'N/A')}\n\n"
            f"Sentiment:\n{report_sections.get('sentiment', 'N/A')}\n\n"
            f"Conflicts:\n{conflict_text or 'None detected'}\n\n"
            f"Respond in JSON (no markdown):\n"
            f'{{"executive_summary": "<3-4 sentence overview>", '
            f'"key_insights": ["insight1", "insight2", "insight3"], '
            f'"risk_factors": ["risk1", "risk2"], '
            f'"recommendations": ["rec1", "rec2", "rec3"]}}'
        )

        raw = llm_call(prompt, f"Synthesise intelligence report for {company_name}", temperature=0.4)
        data = _parse_json(raw)
        elapsed = (time.time() - start) * 1000

        # Calculate overall confidence
        confidences = [r.confidence for r in reports]
        overall_conf = sum(confidences) / len(confidences) if confidences else 0.5

        total_time = sum(r.processing_time_ms for r in reports) + elapsed

        return SynthesisReport(
            company_name=company_name,
            executive_summary=str(data.get("executive_summary", "Report synthesis completed.")),
            market_analysis=report_sections.get("market", "N/A"),
            product_analysis=report_sections.get("product", "N/A"),
            technology_analysis=report_sections.get("tech_stack", "N/A"),
            news_summary=report_sections.get("news", "N/A"),
            sentiment_analysis=report_sections.get("sentiment", "N/A"),
            conflicts_detected=[c.model_dump() for c in conflicts],
            overall_confidence=round(overall_conf, 3),
            key_insights=_safe_list(data.get("key_insights", [])),
            risk_factors=_safe_list(data.get("risk_factors", [])),
            recommendations=_safe_list(data.get("recommendations", [])),
            agent_reports=[r.model_dump() for r in reports],
            total_processing_time_ms=round(total_time, 1),
            agents_used=len(reports),
        )
