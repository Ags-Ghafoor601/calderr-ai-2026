"""
AI-Powered Hiring Pipeline — Data Models
==========================================
Pydantic models for candidates, job descriptions, bias reports, audit logs,
interview questions, and all pipeline state types.
"""

from enum import Enum
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CandidateStatus(str, Enum):
    """Status of a candidate in the hiring pipeline."""
    INGESTED = "ingested"
    SCORED = "scored"
    BIAS_CHECKED = "bias_checked"
    SHORTLISTED = "shortlisted"
    NOT_SHORTLISTED = "not_shortlisted"
    QUESTIONS_GENERATED = "questions_generated"
    PENDING_REVIEW = "pending_review"
    HIRED = "hired"
    REJECTED = "rejected"


class BiasCategory(str, Enum):
    """Categories of bias that can be detected."""
    GENDER = "gender"
    AGE = "age"
    ETHNICITY = "ethnicity"
    EDUCATION_PRESTIGE = "education_prestige"
    NAME_BIAS = "name_bias"
    NONE = "none"


class DecisionType(str, Enum):
    """Type of hiring decision."""
    AUTO = "auto"
    HUMAN = "human"


# ---------------------------------------------------------------------------
# Core Models
# ---------------------------------------------------------------------------

class Candidate(BaseModel):
    """A job candidate with resume information."""
    id: str = Field(default="")
    name: str = Field(...)
    email: str = Field(...)
    phone: str = Field(default="")
    years_experience: int = Field(default=0, ge=0)
    education: str = Field(default="")
    university: str = Field(default="")
    skills: list[str] = Field(default_factory=list)
    previous_roles: list[str] = Field(default_factory=list)
    summary: str = Field(default="")
    resume_text: str = Field(default="")


class JobDescription(BaseModel):
    """A job posting with requirements."""
    id: str = Field(default="")
    title: str = Field(...)
    department: str = Field(default="Engineering")
    required_skills: list[str] = Field(default_factory=list)
    preferred_skills: list[str] = Field(default_factory=list)
    min_experience: int = Field(default=0, ge=0)
    education_requirement: str = Field(default="Bachelor's")
    description: str = Field(default="")
    salary_range: str = Field(default="")


class CandidateScore(BaseModel):
    """Scoring result for a candidate against a job description."""
    candidate_id: str = Field(default="")
    overall_score: float = Field(default=0.0, ge=0.0, le=100.0)
    skills_match: float = Field(default=0.0, ge=0.0, le=100.0)
    experience_match: float = Field(default=0.0, ge=0.0, le=100.0)
    education_match: float = Field(default=0.0, ge=0.0, le=100.0)
    summary_relevance: float = Field(default=0.0, ge=0.0, le=100.0)
    scoring_rationale: str = Field(default="")


class BiasFlag(BaseModel):
    """A detected bias indicator."""
    category: BiasCategory = Field(default=BiasCategory.NONE)
    severity: str = Field(default="low")  # low, medium, high
    description: str = Field(default="")
    recommendation: str = Field(default="")


class BiasReport(BaseModel):
    """Complete bias analysis report for a candidate."""
    candidate_id: str = Field(default="")
    flags: list[BiasFlag] = Field(default_factory=list)
    overall_risk: str = Field(default="low")  # low, medium, high
    adjusted_score: float = Field(default=0.0)
    notes: str = Field(default="")


class InterviewQuestion(BaseModel):
    """A generated interview question."""
    category: str = Field(default="technical")  # technical, behavioral, situational
    question: str = Field(default="")
    follow_up: str = Field(default="")
    evaluation_criteria: str = Field(default="")


class AuditLogEntry(BaseModel):
    """An entry in the audit trail."""
    timestamp: str = Field(default="")
    candidate_id: str = Field(default="")
    action: str = Field(default="")
    details: str = Field(default="")
    decision_by: str = Field(default="system")
    node_name: str = Field(default="")


class HiringDecision(BaseModel):
    """Final hiring decision for a candidate."""
    candidate_id: str = Field(default="")
    decision: str = Field(default="")  # hire, reject, hold
    decided_by: DecisionType = Field(default=DecisionType.AUTO)
    rationale: str = Field(default="")
    interview_questions: list[InterviewQuestion] = Field(default_factory=list)
    bias_report: Optional[BiasReport] = Field(default=None)
    score: Optional[CandidateScore] = Field(default=None)
