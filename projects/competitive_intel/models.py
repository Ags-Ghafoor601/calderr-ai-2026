"""
CalderR Internship – Week 5, Project 5-I-A
=============================================
Competitive Intelligence Agent — Pydantic Models

All typed schemas for inter-agent communication,
report structures, and API responses.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Any
import uuid

from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════════════════════════
#  ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class Priority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    DEGRADED = "degraded"


class ConflictSeverity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class Sentiment(str, Enum):
    VERY_POSITIVE = "very_positive"
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    VERY_NEGATIVE = "very_negative"


# ═══════════════════════════════════════════════════════════════════════════
#  INTER-AGENT MESSAGE SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════

class ResearchRequest(BaseModel):
    """Request from Orchestrator to a specialist agent."""
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    company_name: str = Field(..., min_length=1)
    research_focus: str = Field(..., min_length=5)
    priority: Priority = Field(default=Priority.HIGH)
    context: dict[str, Any] = Field(default_factory=dict)


class AgentReport(BaseModel):
    """Base report from any specialist agent."""
    agent_name: str
    agent_type: str
    request_id: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    company_name: str
    findings: str
    confidence: float = Field(ge=0.0, le=1.0)
    key_data_points: list[str] = Field(default_factory=list)
    processing_time_ms: float = Field(ge=0.0)
    status: AgentStatus = AgentStatus.SUCCESS


# ═══════════════════════════════════════════════════════════════════════════
#  SPECIALIST REPORT SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════

class MarketReport(AgentReport):
    """Report from the Market Agent."""
    agent_type: str = "market"
    market_size_estimate: str = ""
    market_position: str = ""
    growth_trajectory: str = ""
    key_competitors: list[str] = Field(default_factory=list)


class ProductReport(AgentReport):
    """Report from the Product Agent."""
    agent_type: str = "product"
    core_products: list[str] = Field(default_factory=list)
    differentiators: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)


class TechStackReport(AgentReport):
    """Report from the Tech Stack Agent."""
    agent_type: str = "tech_stack"
    inferred_technologies: list[str] = Field(default_factory=list)
    tech_strengths: list[str] = Field(default_factory=list)
    tech_risks: list[str] = Field(default_factory=list)


class NewsReport(AgentReport):
    """Report from the News Agent."""
    agent_type: str = "news"
    recent_developments: list[str] = Field(default_factory=list)
    notable_events: list[str] = Field(default_factory=list)
    news_sentiment: Sentiment = Sentiment.NEUTRAL


class SentimentReport(AgentReport):
    """Report from the Sentiment Agent."""
    agent_type: str = "sentiment"
    overall_sentiment: Sentiment = Sentiment.NEUTRAL
    sentiment_drivers: list[str] = Field(default_factory=list)
    risk_signals: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
#  CONFLICT & SYNTHESIS SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════

class Conflict(BaseModel):
    """A detected conflict between two agent reports."""
    conflict_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    agent_a: str
    agent_b: str
    topic: str
    claim_a: str
    claim_b: str
    severity: ConflictSeverity = ConflictSeverity.MEDIUM
    resolution: str = ""
    resolved: bool = False


class SynthesisReport(BaseModel):
    """Final synthesised intelligence briefing."""
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    company_name: str
    executive_summary: str
    market_analysis: str
    product_analysis: str
    technology_analysis: str
    news_summary: str
    sentiment_analysis: str
    conflicts_detected: list[dict] = Field(default_factory=list)
    overall_confidence: float = Field(ge=0.0, le=1.0)
    key_insights: list[str] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    agent_reports: list[dict] = Field(default_factory=list)
    total_processing_time_ms: float = 0.0
    agents_used: int = 0


# ═══════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR STATE
# ═══════════════════════════════════════════════════════════════════════════

class OrchestratorState(BaseModel):
    """Full state for the orchestrator workflow."""
    company_name: str = ""
    research_requests: list[dict] = Field(default_factory=list)
    agent_reports: list[dict] = Field(default_factory=list)
    conflicts: list[dict] = Field(default_factory=list)
    synthesis: dict = Field(default_factory=dict)
    agent_statuses: dict[str, str] = Field(default_factory=dict)
    total_time_ms: float = 0.0
    status: str = "pending"


# ═══════════════════════════════════════════════════════════════════════════
#  API RESPONSE SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    """API health check response."""
    status: str = "healthy"
    agents: int = 5
    version: str = "1.0.0"


class AnalysisResponse(BaseModel):
    """API response for a company analysis request."""
    status: str
    company: str
    report: Optional[dict] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0
