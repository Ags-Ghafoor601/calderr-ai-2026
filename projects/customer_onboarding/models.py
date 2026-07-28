"""
Customer Onboarding Agent — Pydantic Models
=============================================
Data models for the customer onboarding workflow.
Defines customer profiles, account types, and validation rules.
"""

from enum import Enum
from typing import Optional
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
import re


class AccountTier(str, Enum):
    """Account tier classification based on company size and revenue."""
    STARTER = "starter"          # Small business, < 10 employees
    PROFESSIONAL = "professional"  # Medium business, 10-100 employees
    ENTERPRISE = "enterprise"    # Large business, > 100 employees or > $1M revenue


class OnboardingStatus(str, Enum):
    """Status of the onboarding process."""
    PENDING = "pending"
    INFO_COLLECTED = "info_collected"
    VALIDATED = "validated"
    AWAITING_APPROVAL = "awaiting_approval"
    APPROVED = "approved"
    REJECTED = "rejected"
    ACCOUNT_CREATED = "account_created"
    NOTIFIED = "notified"
    FOLLOWUP_SCHEDULED = "followup_scheduled"
    COMPLETED = "completed"


class CustomerInfo(BaseModel):
    """Customer information collected during onboarding."""
    company_name: str = Field(..., min_length=2, max_length=200)
    contact_name: str = Field(..., min_length=2, max_length=100)
    contact_email: str = Field(...)
    phone: str = Field(default="")
    company_size: int = Field(..., ge=1)
    annual_revenue: float = Field(default=0.0, ge=0)
    industry: str = Field(default="Technology")
    use_case: str = Field(default="General")

    @field_validator("contact_email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Validate email format."""
        pattern = r"^[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+$"
        if not re.match(pattern, v):
            raise ValueError(f"Invalid email format: {v}")
        return v.lower()


class AccountDetails(BaseModel):
    """Generated account details after approval."""
    account_id: str = Field(default="")
    tier: AccountTier = Field(default=AccountTier.STARTER)
    api_key: str = Field(default="")
    created_at: str = Field(default="")
    welcome_sent: bool = Field(default=False)
    followup_date: str = Field(default="")


class ValidationResult(BaseModel):
    """Result of customer information validation."""
    is_valid: bool = Field(default=False)
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    requires_human_approval: bool = Field(default=False)
    approval_reason: str = Field(default="")


# ---------------------------------------------------------------------------
# Sample Customer Data for Demo
# ---------------------------------------------------------------------------

SAMPLE_CUSTOMERS = [
    {
        "company_name": "TechStart Inc.",
        "contact_name": "Alice Johnson",
        "contact_email": "alice@techstart.com",
        "phone": "+1-555-0101",
        "company_size": 5,
        "annual_revenue": 250000,
        "industry": "Technology",
        "use_case": "API integration for data analytics",
    },
    {
        "company_name": "MegaCorp Global Solutions",
        "contact_name": "Robert Chen",
        "contact_email": "r.chen@megacorp.com",
        "phone": "+1-555-0202",
        "company_size": 500,
        "annual_revenue": 5000000,
        "industry": "Finance",
        "use_case": "Enterprise AI workflow automation",
    },
    {
        "company_name": "GreenLeaf Analytics",
        "contact_name": "Sarah Williams",
        "contact_email": "sarah@greenleaf.io",
        "phone": "+1-555-0303",
        "company_size": 45,
        "annual_revenue": 800000,
        "industry": "Healthcare",
        "use_case": "Patient data processing pipeline",
    },
    {
        "company_name": "DataFlow Systems",
        "contact_name": "James Park",
        "contact_email": "j.park@dataflow.dev",
        "phone": "+1-555-0404",
        "company_size": 120,
        "annual_revenue": 2500000,
        "industry": "Technology",
        "use_case": "Real-time data ingestion platform",
    },
    {
        "company_name": "",  # Invalid — empty name
        "contact_name": "Test User",
        "contact_email": "invalid-email",  # Invalid email
        "phone": "",
        "company_size": 0,  # Invalid — must be >= 1
        "annual_revenue": -100,  # Invalid — negative
        "industry": "Unknown",
        "use_case": "Testing validation",
    },
]
