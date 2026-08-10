"""
Enterprise AI Memory Platform — Pydantic Models
=================================================
All typed schemas for the 4 memory types, API requests/responses,
tenant management, and consolidation records.
"""

import uuid
from datetime import datetime, timezone
from typing import Optional, Any
from enum import Enum
from pydantic import BaseModel, Field, field_validator


# ═══════════════════════════════════════════════════════════════════════════
#  ENUMS
# ═══════════════════════════════════════════════════════════════════════════

class MemoryType(str, Enum):
    """The four core memory types in the platform."""
    EPISODIC = "episodic"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    KNOWLEDGE_GRAPH = "knowledge_graph"


class RuleDomain(str, Enum):
    """Domain categories for procedural correction rules."""
    FACTUAL = "factual"
    FORMATTING = "formatting"
    TONE = "tone"
    REASONING = "reasoning"
    ACCURACY = "accuracy"
    COMPLETENESS = "completeness"
    GENERAL = "general"


class ConsolidationStatus(str, Enum):
    """Status of a consolidation operation."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


# ═══════════════════════════════════════════════════════════════════════════
#  TENANT MODEL
# ═══════════════════════════════════════════════════════════════════════════

class Tenant(BaseModel):
    """A tenant (user or organisation) with isolated memory namespaces."""
    tenant_id: str = Field(..., min_length=1, description="Unique tenant identifier")
    name: str = Field(..., min_length=1, description="Human-readable tenant name")
    created_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    is_active: bool = Field(default=True)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @field_validator("tenant_id")
    @classmethod
    def validate_tenant_id(cls, v: str) -> str:
        if " " in v.strip():
            raise ValueError("Tenant ID must not contain spaces")
        return v.strip().lower()


# ═══════════════════════════════════════════════════════════════════════════
#  EPISODIC MEMORY MODELS
# ═══════════════════════════════════════════════════════════════════════════

class EpisodicMemoryCreate(BaseModel):
    """Request to store a new episodic memory."""
    session_id: str = Field(..., min_length=1, description="Session this memory belongs to")
    content: str = Field(..., min_length=1, description="The interaction content")
    role: str = Field(default="user", description="Role: user or assistant")
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodicMemory(BaseModel):
    """A stored episodic memory entry."""
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    tenant_id: str = Field(...)
    session_id: str = Field(...)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    content: str = Field(...)
    role: str = Field(default="user")
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0)
    is_consolidated: bool = Field(default=False)
    metadata: dict[str, Any] = Field(default_factory=dict)


class EpisodicQueryRequest(BaseModel):
    """Request to query episodic memories."""
    query: Optional[str] = Field(default=None, description="Semantic query text")
    session_id: Optional[str] = Field(default=None, description="Filter by session")
    limit: int = Field(default=10, ge=1, le=100)
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0)


# ═══════════════════════════════════════════════════════════════════════════
#  SEMANTIC MEMORY MODELS
# ═══════════════════════════════════════════════════════════════════════════

class SemanticMemoryCreate(BaseModel):
    """Request to store a semantic fact or preference."""
    fact: str = Field(..., min_length=1, description="The fact or knowledge to store")
    category: str = Field(default="general", description="Category: preference, fact, profile, knowledge")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticMemory(BaseModel):
    """A stored semantic memory entry."""
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    tenant_id: str = Field(...)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    fact: str = Field(...)
    category: str = Field(default="general")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SemanticQueryRequest(BaseModel):
    """Request to query semantic memories."""
    query: str = Field(..., min_length=1, description="Semantic search query")
    category: Optional[str] = Field(default=None, description="Filter by category")
    limit: int = Field(default=5, ge=1, le=50)


# ═══════════════════════════════════════════════════════════════════════════
#  PROCEDURAL MEMORY MODELS
# ═══════════════════════════════════════════════════════════════════════════

class ProceduralRuleCreate(BaseModel):
    """Request to store a procedural correction rule."""
    original_mistake: str = Field(..., min_length=1)
    correction: str = Field(..., min_length=1)
    rule_text: str = Field(..., min_length=5, description="Generalised rule to prevent the mistake")
    domain: RuleDomain = Field(default=RuleDomain.GENERAL)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)


class ProceduralRule(BaseModel):
    """A stored procedural correction rule."""
    rule_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    tenant_id: str = Field(...)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    original_mistake: str = Field(...)
    correction: str = Field(...)
    rule_text: str = Field(...)
    domain: RuleDomain = Field(default=RuleDomain.GENERAL)
    confidence: float = Field(default=0.7, ge=0.0, le=1.0)
    application_count: int = Field(default=0)
    last_applied: Optional[str] = Field(default=None)
    is_active: bool = Field(default=True)


class ProceduralQueryRequest(BaseModel):
    """Request to query procedural rules."""
    query: Optional[str] = Field(default=None, description="Semantic query for relevant rules")
    domain: Optional[RuleDomain] = Field(default=None, description="Filter by domain")
    active_only: bool = Field(default=True)
    limit: int = Field(default=10, ge=1, le=50)


# ═══════════════════════════════════════════════════════════════════════════
#  KNOWLEDGE GRAPH MODELS
# ═══════════════════════════════════════════════════════════════════════════

class EntityCreate(BaseModel):
    """Request to add an entity to the knowledge graph."""
    name: str = Field(..., min_length=1)
    entity_type: str = Field(default="concept", description="person, company, concept, place, etc.")
    description: str = Field(default="")
    aliases: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RelationshipCreate(BaseModel):
    """Request to add a relationship to the knowledge graph."""
    source: str = Field(..., min_length=1, description="Source entity name")
    target: str = Field(..., min_length=1, description="Target entity name")
    relation_type: str = Field(..., min_length=1, description="Type of relationship")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    evidence: str = Field(default="")


class GraphQueryRequest(BaseModel):
    """Request to query the knowledge graph."""
    entity: Optional[str] = Field(default=None, description="Entity to explore")
    depth: int = Field(default=2, ge=1, le=5)
    query: Optional[str] = Field(default=None, description="Natural language query")


class GraphEntity(BaseModel):
    """An entity in the knowledge graph."""
    name: str = Field(...)
    entity_type: str = Field(default="concept")
    description: str = Field(default="")
    aliases: list[str] = Field(default_factory=list)
    connections: int = Field(default=0)


class GraphRelationship(BaseModel):
    """A relationship in the knowledge graph."""
    source: str = Field(...)
    target: str = Field(...)
    relation_type: str = Field(...)
    confidence: float = Field(default=0.8)


class GraphStats(BaseModel):
    """Statistics for a knowledge graph."""
    tenant_id: str = Field(...)
    total_entities: int = Field(default=0)
    total_relationships: int = Field(default=0)
    entity_types: dict[str, int] = Field(default_factory=dict)
    relationship_types: dict[str, int] = Field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════
#  CONSOLIDATION MODELS
# ═══════════════════════════════════════════════════════════════════════════

class ConsolidationRecord(BaseModel):
    """Record of a consolidation operation."""
    consolidation_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:12])
    tenant_id: str = Field(...)
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: ConsolidationStatus = Field(default=ConsolidationStatus.PENDING)
    episodes_processed: int = Field(default=0)
    episodes_consolidated: int = Field(default=0)
    summary_text: str = Field(default="")
    rules_promoted: int = Field(default=0)
    memories_pruned: int = Field(default=0)


class ConsolidationConfig(BaseModel):
    """Configuration for the consolidation worker."""
    episode_threshold: int = Field(default=100, ge=10, description="Trigger consolidation after N episodes")
    batch_size: int = Field(default=50, ge=5, description="Episodes per consolidation batch")
    min_importance_to_keep: float = Field(default=0.3, ge=0.0, le=1.0)
    rule_confidence_promotion: float = Field(default=0.85, ge=0.0, le=1.0)


# ═══════════════════════════════════════════════════════════════════════════
#  API RESPONSE MODELS
# ═══════════════════════════════════════════════════════════════════════════

class MemoryResponse(BaseModel):
    """Standard API response for memory operations."""
    success: bool = Field(default=True)
    message: str = Field(default="")
    data: Any = Field(default=None)
    count: int = Field(default=0)


class TenantStats(BaseModel):
    """Statistics for a tenant's memory stores."""
    tenant_id: str = Field(...)
    tenant_name: str = Field(default="")
    episodic_count: int = Field(default=0)
    semantic_count: int = Field(default=0)
    procedural_count: int = Field(default=0)
    graph_entities: int = Field(default=0)
    graph_relationships: int = Field(default=0)
    total_memories: int = Field(default=0)
    last_activity: Optional[str] = Field(default=None)
    consolidations_run: int = Field(default=0)


class PlatformStats(BaseModel):
    """Platform-wide statistics."""
    total_tenants: int = Field(default=0)
    active_tenants: int = Field(default=0)
    total_memories: int = Field(default=0)
    total_rules: int = Field(default=0)
    total_graph_entities: int = Field(default=0)
    total_consolidations: int = Field(default=0)
    tenants: list[TenantStats] = Field(default_factory=list)
