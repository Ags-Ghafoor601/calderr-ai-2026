"""
Pydantic Models for Document Processing Pipeline
===================================================
All structured output models used by the extraction agent.
These models validate the LLM's extraction results and define
the database schema.
"""

from datetime import datetime, date
from typing import Optional
from enum import Enum

from pydantic import BaseModel, Field, field_validator


class DocumentType(str, Enum):
    """Supported document types."""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"


class SeverityLevel(str, Enum):
    """Priority/severity for action items."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class EntityType(str, Enum):
    """Types of named entities."""
    PERSON = "person"
    ORGANIZATION = "organization"
    LOCATION = "location"
    DATE = "date"
    MONEY = "money"
    EMAIL = "email"
    PHONE = "phone"
    URL = "url"
    SOFTWARE = "software"
    TECHNOLOGY = "technology"
    PRODUCT = "product"
    COURSE = "course"
    OTHER = "other"


class ExtractedEntity(BaseModel):
    """A single named entity extracted from the document."""
    text: str = Field(description="The entity text as it appears in the document")
    entity_type: EntityType = Field(description="The type of entity")
    confidence: Optional[float] = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Confidence score for this extraction (0.0–1.0)",
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_empty(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("Entity text must not be empty")
        return v.strip()


class ExtractedDate(BaseModel):
    """A date or deadline extracted from the document."""
    text: str = Field(description="The date as written in the document")
    normalized: Optional[str] = Field(
        default=None,
        description="ISO-format date if parseable (YYYY-MM-DD)",
    )
    context: Optional[str] = Field(
        default="",
        description="Brief context about what this date refers to",
    )


class ActionItem(BaseModel):
    """An action item or task extracted from the document."""
    description: str = Field(description="What needs to be done")
    assignee: Optional[str] = Field(
        default=None,
        description="Who is responsible (if mentioned)",
    )
    deadline: Optional[str] = Field(
        default=None,
        description="When it's due (if mentioned)",
    )
    severity: Optional[SeverityLevel] = Field(
        default=SeverityLevel.MEDIUM,
        description="Priority level of this action item",
    )


class DocumentExtraction(BaseModel):
    """
    Complete structured extraction from a document.
    This is what the LLM extraction agent returns.
    """
    summary: str = Field(
        description="A concise 2-3 sentence summary of the document's main content"
    )
    key_terms: list[str] = Field(
        default_factory=list,
        description="Important terms, concepts, or keywords from the document (Limit to top 20 most important)",
    )
    entities: list[ExtractedEntity] = Field(
        default_factory=list,
        description="Named entities found in the document (Limit to top 15 most important)",
    )
    dates: list[ExtractedDate] = Field(
        default_factory=list,
        description="Dates and deadlines mentioned in the document (Limit to top 10)",
    )
    action_items: list[ActionItem] = Field(
        default_factory=list,
        description="Action items, tasks, or to-dos found in the document (Limit to top 10 most critical)",
    )
    document_type_guess: str = Field(
        default="general",
        description="Best guess of document type (e.g., 'contract', 'report', 'email', 'memo', 'invoice')",
    )
    language: str = Field(
        default="English",
        description="Primary language of the document",
    )
    word_count: Optional[int] = Field(
        default=None,
        description="Approximate word count of the source document",
    )

    @field_validator("key_terms")
    @classmethod
    def deduplicate_terms(cls, v: list[str]) -> list[str]:
        seen = set()
        result = []
        for term in v:
            t = term.strip()
            if t and t.lower() not in seen:
                seen.add(t.lower())
                result.append(t)
        return result


class DocumentRecord(BaseModel):
    """
    Full document record stored in the database.
    Combines file metadata with extraction results.
    """
    id: Optional[int] = Field(default=None, description="Database ID (auto-generated)")
    filename: str = Field(description="Original filename")
    file_type: DocumentType = Field(description="Document type (pdf, docx, txt)")
    file_size_bytes: int = Field(description="File size in bytes")
    uploaded_at: str = Field(
        default_factory=lambda: datetime.now().isoformat(),
        description="Upload timestamp (ISO format)",
    )
    extraction: DocumentExtraction = Field(description="Structured extraction results")
    processing_time_ms: float = Field(
        default=0.0,
        description="Time taken to process this document (milliseconds)",
    )
    status: str = Field(default="completed", description="Processing status")
    error: Optional[str] = Field(default=None, description="Error message if processing failed")
