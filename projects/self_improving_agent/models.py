"""
Procedural Memory & Self-Improving Agent — Pydantic Models
==========================================================
All data schemas for the self-improving agent system.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, Any
from enum import Enum
from pydantic import BaseModel, Field


class RuleDomain(str, Enum):
    """Domain categories for correction rules."""
    FACTUAL = "factual"
    FORMATTING = "formatting"
    TONE = "tone"
    REASONING = "reasoning"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    GENERAL = "general"


class CorrectionRule(BaseModel):
    """A procedural memory rule extracted from a user correction."""
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    original_mistake: str = Field(..., description="What the agent did wrong")
    correction: str = Field(..., description="What the user corrected it to")
    rule_text: str = Field(..., description="Generalised rule to prevent this mistake")
    domain: RuleDomain = Field(default=RuleDomain.GENERAL)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    application_count: int = Field(default=0, ge=0)
    last_applied: Optional[str] = Field(default=None)
    source_interaction_id: str = Field(default="")
    is_active: bool = Field(default=True)


class InteractionLog(BaseModel):
    """A single interaction between user and agent."""
    interaction_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    user_message: str = Field(...)
    agent_response: str = Field(...)
    was_corrected: bool = Field(default=False)
    correction_text: Optional[str] = Field(default=None)
    rules_applied: list[str] = Field(default_factory=list, description="Rule IDs that were applied")
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class PerformanceRecord(BaseModel):
    """Tracks agent performance over time."""
    record_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    interaction_number: int = Field(..., ge=1)
    was_correct: bool = Field(default=True)
    error_type: Optional[str] = Field(default=None)
    rules_applied_count: int = Field(default=0)
    total_rules_available: int = Field(default=0)
    quality_score: float = Field(default=0.5, ge=0.0, le=1.0)


class LearningCurvePoint(BaseModel):
    """A single point on the learning curve."""
    interaction_number: int = Field(...)
    cumulative_accuracy: float = Field(..., ge=0.0, le=1.0)
    rolling_accuracy: float = Field(..., ge=0.0, le=1.0)
    total_rules: int = Field(default=0)
    error_rate: float = Field(default=0.0, ge=0.0, le=1.0)


class AgentState(BaseModel):
    """Complete state of the self-improving agent."""
    total_interactions: int = Field(default=0)
    total_corrections: int = Field(default=0)
    total_rules: int = Field(default=0)
    current_accuracy: float = Field(default=0.0, ge=0.0, le=1.0)
    improvement_percentage: float = Field(default=0.0)
    learning_curve: list[LearningCurvePoint] = Field(default_factory=list)
