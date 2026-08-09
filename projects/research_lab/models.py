"""
CalderR Internship – Week 5, Project 5-P-A
=============================================
Autonomous AI Research Lab — Pydantic Models

All typed schemas for multi-phase research orchestration,
domain classification, agent handoffs, and report structures.
"""

from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Any
import uuid

from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════════════════════════
#  ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class ResearchPhase(str, Enum):
    """The 5 sequential phases of the research pipeline."""
    HYPOTHESIS = "hypothesis"
    EVIDENCE = "evidence"
    CRITIQUE = "critique"
    SYNTHESIS = "synthesis"
    PEER_REVIEW = "peer_review"


class ResearchDomain(str, Enum):
    """Supported research domains — each triggers different specialist agents."""
    TECHNOLOGY = "technology"
    MEDICINE = "medicine"
    ECONOMICS = "economics"
    ENVIRONMENT = "environment"
    SOCIAL_SCIENCE = "social_science"
    GENERAL = "general"


class AgentRole(str, Enum):
    """All possible agent roles in the research lab."""
    ORCHESTRATOR = "orchestrator"
    DOMAIN_CLASSIFIER = "domain_classifier"
    HYPOTHESIS_GENERATOR = "hypothesis_generator"
    LITERATURE_REVIEWER = "literature_reviewer"
    DATA_ANALYST = "data_analyst"
    METHODOLOGY_EXPERT = "methodology_expert"
    DOMAIN_SPECIALIST = "domain_specialist"
    CRITIC = "critic"
    SYNTHESISER = "synthesiser"
    PEER_REVIEWER = "peer_reviewer"


class AgentStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    DEGRADED = "degraded"


class Severity(str, Enum):
    NONE = "none"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class Verdict(str, Enum):
    """Peer review verdict."""
    ACCEPT = "accept"
    MINOR_REVISIONS = "minor_revisions"
    MAJOR_REVISIONS = "major_revisions"
    REJECT = "reject"


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 1: HYPOTHESIS GENERATION
# ═══════════════════════════════════════════════════════════════════════════

class Hypothesis(BaseModel):
    """A single generated hypothesis."""
    hypothesis_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    statement: str = Field(..., min_length=10)
    rationale: str = Field(..., min_length=10)
    testability: str = Field(default="", description="How this hypothesis can be tested")
    novelty_score: float = Field(default=0.5, ge=0.0, le=1.0)
    domain_relevance: float = Field(default=0.5, ge=0.0, le=1.0)


class HypothesisReport(BaseModel):
    """Output from Phase 1."""
    phase: ResearchPhase = ResearchPhase.HYPOTHESIS
    topic: str
    domain: ResearchDomain
    hypotheses: list[Hypothesis]
    methodology_suggestion: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    processing_time_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 2: EVIDENCE GATHERING
# ═══════════════════════════════════════════════════════════════════════════

class EvidenceItem(BaseModel):
    """A single piece of evidence supporting or opposing a hypothesis."""
    evidence_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    hypothesis_id: str = Field(..., description="Which hypothesis this evidence relates to")
    source_type: str = Field(default="literature", description="literature/data/expert_opinion")
    summary: str = Field(..., min_length=10)
    supports_hypothesis: bool = True
    strength: float = Field(default=0.5, ge=0.0, le=1.0,
                            description="How strongly this evidence supports/opposes")
    source_reference: str = Field(default="")


class EvidenceReport(BaseModel):
    """Output from Phase 2."""
    phase: ResearchPhase = ResearchPhase.EVIDENCE
    topic: str
    evidence_items: list[EvidenceItem]
    literature_summary: str = ""
    data_analysis_summary: str = ""
    methodology_notes: str = ""
    agents_used: list[str] = Field(default_factory=list)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    processing_time_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 3: CRITICAL ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

class Critique(BaseModel):
    """A critique of a specific hypothesis or evidence item."""
    critique_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    target_type: str = Field(default="hypothesis", description="hypothesis/evidence/methodology")
    target_id: str = Field(default="")
    issue: str = Field(..., min_length=10)
    severity: Severity = Severity.MEDIUM
    suggestion: str = Field(default="")
    addressed: bool = False


class CritiqueReport(BaseModel):
    """Output from Phase 3."""
    phase: ResearchPhase = ResearchPhase.CRITIQUE
    topic: str
    critiques: list[Critique]
    methodology_weaknesses: list[str] = Field(default_factory=list)
    bias_warnings: list[str] = Field(default_factory=list)
    overall_rigor_score: float = Field(default=0.5, ge=0.0, le=1.0)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    processing_time_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 4: SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════════

class SynthesisReport(BaseModel):
    """Output from Phase 4."""
    phase: ResearchPhase = ResearchPhase.SYNTHESIS
    topic: str
    abstract: str = Field(..., min_length=20)
    introduction: str = ""
    methodology: str = ""
    findings: str = ""
    discussion: str = ""
    conclusion: str = ""
    limitations: str = ""
    future_work: str = ""
    key_contributions: list[str] = Field(default_factory=list)
    overall_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    processing_time_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  PHASE 5: PEER REVIEW
# ═══════════════════════════════════════════════════════════════════════════

class ReviewComment(BaseModel):
    """A single peer review comment."""
    comment_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    section: str = Field(default="general")
    comment: str = Field(..., min_length=10)
    severity: Severity = Severity.MEDIUM
    actionable: bool = True


class PeerReviewReport(BaseModel):
    """Output from Phase 5."""
    phase: ResearchPhase = ResearchPhase.PEER_REVIEW
    topic: str
    reviewer_name: str = "peer-reviewer"
    verdict: Verdict = Verdict.MINOR_REVISIONS
    overall_score: float = Field(default=0.5, ge=0.0, le=1.0,
                                  description="1.0 = publishable, 0.0 = reject")
    comments: list[ReviewComment] = Field(default_factory=list)
    strengths: list[str] = Field(default_factory=list)
    weaknesses: list[str] = Field(default_factory=list)
    recommendation: str = ""
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    processing_time_ms: float = 0.0


# ═══════════════════════════════════════════════════════════════════════════
#  FULL RESEARCH REPORT (AGGREGATED)
# ═══════════════════════════════════════════════════════════════════════════

class FullResearchReport(BaseModel):
    """The complete research output — all 5 phases aggregated."""
    report_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    topic: str
    domain: ResearchDomain
    hypothesis_report: dict = Field(default_factory=dict)
    evidence_report: dict = Field(default_factory=dict)
    critique_report: dict = Field(default_factory=dict)
    synthesis_report: dict = Field(default_factory=dict)
    peer_review_report: dict = Field(default_factory=dict)
    agents_assembled: list[str] = Field(default_factory=list)
    total_agents_used: int = 0
    total_processing_time_ms: float = 0.0
    phases_completed: int = 0
    overall_quality_score: float = Field(default=0.0, ge=0.0, le=1.0)
    status: str = "pending"
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


# ═══════════════════════════════════════════════════════════════════════════
#  ORCHESTRATOR STATE
# ═══════════════════════════════════════════════════════════════════════════

class PipelineState(BaseModel):
    """Full state for the 5-phase research pipeline."""
    topic: str = ""
    domain: ResearchDomain = ResearchDomain.GENERAL
    assembled_agents: list[str] = Field(default_factory=list)
    current_phase: ResearchPhase = ResearchPhase.HYPOTHESIS
    phase_results: dict[str, dict] = Field(default_factory=dict)
    agent_logs: list[dict] = Field(default_factory=list)
    total_time_ms: float = 0.0
    status: str = "pending"


# ═══════════════════════════════════════════════════════════════════════════
#  API SCHEMAS
# ═══════════════════════════════════════════════════════════════════════════

class HealthResponse(BaseModel):
    status: str = "healthy"
    project: str = "Autonomous AI Research Lab"
    version: str = "1.0.0"
    phases: int = 5


class ResearchRequest(BaseModel):
    topic: str = Field(..., min_length=5)
    domain: Optional[str] = Field(default=None, description="Auto-detected if not provided")


class ResearchResponse(BaseModel):
    status: str
    topic: str
    domain: str = ""
    report: Optional[dict] = None
    error: Optional[str] = None
    processing_time_ms: float = 0.0
