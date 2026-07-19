"""Enterprise Document Intelligence Platform — Pydantic Models."""

from datetime import datetime
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class DocumentStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    READY = "ready"
    FAILED = "failed"


# ---------------------------------------------------------------------------
# Tenant Models
# ---------------------------------------------------------------------------
class TenantCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=100, description="Tenant display name")
    description: str = Field("", max_length=500, description="Tenant description")


class TenantResponse(BaseModel):
    tenant_id: str
    name: str
    description: str
    document_count: int = 0
    chunk_count: int = 0
    created_at: str


# ---------------------------------------------------------------------------
# Document Models
# ---------------------------------------------------------------------------
class DocumentResponse(BaseModel):
    document_id: str
    tenant_id: str
    filename: str
    file_size: int
    status: DocumentStatus
    chunk_count: int = 0
    uploaded_at: str
    processed_at: Optional[str] = None


class DocumentListResponse(BaseModel):
    tenant_id: str
    documents: list[DocumentResponse]
    total: int


# ---------------------------------------------------------------------------
# Query Models
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int = Field(5, ge=1, le=20)
    use_reranking: bool = Field(True, description="Apply cross-encoder re-ranking")
    use_hybrid: bool = Field(True, description="Use hybrid BM25 + semantic search")


class SourceChunk(BaseModel):
    text: str
    source_file: str
    page: int
    similarity_score: float
    chunk_index: int


class QueryResponse(BaseModel):
    question: str
    answer: str
    confidence_score: float = Field(..., ge=0, le=1)
    source_chunks: list[SourceChunk]
    retrieval_method: str
    processing_time_ms: float
    tenant_id: str


# ---------------------------------------------------------------------------
# Stats / Admin Models
# ---------------------------------------------------------------------------
class PlatformStats(BaseModel):
    total_tenants: int
    total_documents: int
    total_chunks: int
    tenants: list[TenantResponse]


class HealthResponse(BaseModel):
    status: str = "healthy"
    version: str
    uptime_seconds: float
