"""
Enterprise AI Memory Platform — Memory Router
===============================================
Central router that decides which memory store(s) to
read/write based on memory type and provides a unified
interface for all 4 memory types.
"""

from pathlib import Path
from typing import Any, Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models import (
    MemoryType, MemoryResponse, TenantStats,
    EpisodicMemoryCreate, EpisodicQueryRequest,
    SemanticMemoryCreate, SemanticQueryRequest,
    ProceduralRuleCreate, ProceduralQueryRequest,
    EntityCreate, RelationshipCreate, GraphQueryRequest,
)

from stores.episodic import EpisodicStore
from stores.semantic import SemanticStore
from stores.procedural import ProceduralStore
from stores.knowledge_graph import KnowledgeGraphStore
from stores.tenants import TenantManager


class MemoryRouter:
    """Central router for all memory operations across all stores and tenants."""

    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        db_path = f"{data_dir}/memory_platform.db"
        chroma_path = f"{data_dir}/chromadb"
        graph_dir = f"{data_dir}/graphs"

        self.episodic = EpisodicStore(db_path=db_path)
        self.semantic = SemanticStore(persist_dir=chroma_path)
        self.procedural = ProceduralStore(db_path=db_path)
        self.graph = KnowledgeGraphStore(data_dir=graph_dir)
        self.tenants = TenantManager(db_path=db_path)

    # ─── Tenant Operations ────────────────────────────────────────

    def ensure_tenant(self, tenant_id: str, name: Optional[str] = None):
        """Ensure tenant exists; create if not."""
        if not self.tenants.tenant_exists(tenant_id):
            self.tenants.create_tenant(tenant_id, name or tenant_id.title())

    # ─── Episodic API ─────────────────────────────────────────────

    def store_episodic(self, tenant_id: str, memory: EpisodicMemoryCreate) -> MemoryResponse:
        self.ensure_tenant(tenant_id)
        entry = self.episodic.store(tenant_id, memory)
        return MemoryResponse(
            success=True,
            message=f"Episodic memory stored: {entry.memory_id}",
            data=entry.model_dump(),
            count=1,
        )

    def query_episodic(self, tenant_id: str, request: EpisodicQueryRequest) -> MemoryResponse:
        self.ensure_tenant(tenant_id)
        results = self.episodic.query(tenant_id, request)
        return MemoryResponse(
            success=True,
            message=f"Found {len(results)} episodic memories",
            data=[r.model_dump() for r in results],
            count=len(results),
        )

    def delete_episodic(self, tenant_id: str, memory_id: str) -> MemoryResponse:
        success = self.episodic.delete(tenant_id, memory_id)
        return MemoryResponse(
            success=success,
            message="Deleted" if success else "Not found",
        )

    # ─── Semantic API ─────────────────────────────────────────────

    def store_semantic(self, tenant_id: str, memory: SemanticMemoryCreate) -> MemoryResponse:
        self.ensure_tenant(tenant_id)
        entry = self.semantic.store(tenant_id, memory)
        return MemoryResponse(
            success=True,
            message=f"Semantic memory stored: {entry.memory_id}",
            data=entry.model_dump(),
            count=1,
        )

    def query_semantic(self, tenant_id: str, request: SemanticQueryRequest) -> MemoryResponse:
        self.ensure_tenant(tenant_id)
        results = self.semantic.query(tenant_id, request)
        return MemoryResponse(
            success=True,
            message=f"Found {len(results)} semantic memories",
            data=[r.model_dump() for r in results],
            count=len(results),
        )

    def delete_semantic(self, tenant_id: str, memory_id: str) -> MemoryResponse:
        success = self.semantic.delete(tenant_id, memory_id)
        return MemoryResponse(
            success=success,
            message="Deleted" if success else "Not found",
        )

    # ─── Procedural API ──────────────────────────────────────────

    def store_procedural(self, tenant_id: str, rule: ProceduralRuleCreate) -> MemoryResponse:
        self.ensure_tenant(tenant_id)
        entry = self.procedural.store(tenant_id, rule)
        return MemoryResponse(
            success=True,
            message=f"Procedural rule stored: {entry.rule_id}",
            data=entry.model_dump(),
            count=1,
        )

    def query_procedural(self, tenant_id: str, request: ProceduralQueryRequest) -> MemoryResponse:
        self.ensure_tenant(tenant_id)
        results = self.procedural.query(tenant_id, request)
        return MemoryResponse(
            success=True,
            message=f"Found {len(results)} procedural rules",
            data=[r.model_dump() for r in results],
            count=len(results),
        )

    def apply_rule(self, tenant_id: str, rule_id: str) -> MemoryResponse:
        self.procedural.increment_application(tenant_id, rule_id)
        return MemoryResponse(success=True, message=f"Rule {rule_id} application incremented")

    def delete_procedural(self, tenant_id: str, rule_id: str) -> MemoryResponse:
        success = self.procedural.delete(tenant_id, rule_id)
        return MemoryResponse(
            success=success,
            message="Deleted" if success else "Not found",
        )

    # ─── Knowledge Graph API ─────────────────────────────────────

    def add_entity(self, tenant_id: str, entity: EntityCreate) -> MemoryResponse:
        self.ensure_tenant(tenant_id)
        result = self.graph.add_entity(tenant_id, entity)
        return MemoryResponse(
            success=True,
            message=f"Entity added: {result.name}",
            data=result.model_dump(),
            count=1,
        )

    def add_relationship(self, tenant_id: str, rel: RelationshipCreate) -> MemoryResponse:
        self.ensure_tenant(tenant_id)
        result = self.graph.add_relationship(tenant_id, rel)
        return MemoryResponse(
            success=True,
            message=f"Relationship added: {result.source} → {result.target}",
            data=result.model_dump(),
            count=1,
        )

    def query_graph(self, tenant_id: str, request: GraphQueryRequest) -> MemoryResponse:
        self.ensure_tenant(tenant_id)
        result = self.graph.query(tenant_id, request)
        return MemoryResponse(
            success=True,
            message="Graph query result",
            data=result,
        )

    def get_graph_stats(self, tenant_id: str) -> MemoryResponse:
        stats = self.graph.get_stats(tenant_id)
        return MemoryResponse(
            success=True,
            message="Graph statistics",
            data=stats.model_dump(),
        )

    def find_path(self, tenant_id: str, source: str, target: str) -> MemoryResponse:
        path = self.graph.find_path(tenant_id, source, target)
        return MemoryResponse(
            success=True if path else False,
            message=f"Path with {len(path)} hops" if path else "No path found",
            data=path,
            count=len(path),
        )

    # ─── Tenant Stats ────────────────────────────────────────────

    def get_tenant_stats(self, tenant_id: str) -> TenantStats:
        """Get comprehensive stats for a single tenant."""
        tenant = self.tenants.get_tenant(tenant_id)
        return TenantStats(
            tenant_id=tenant_id,
            tenant_name=tenant.name if tenant else tenant_id,
            episodic_count=self.episodic.count(tenant_id),
            semantic_count=self.semantic.count(tenant_id),
            procedural_count=self.procedural.count(tenant_id),
            graph_entities=self.graph.entity_count(tenant_id),
            graph_relationships=self.graph.relationship_count(tenant_id),
            total_memories=(
                self.episodic.count(tenant_id) +
                self.semantic.count(tenant_id) +
                self.procedural.count(tenant_id) +
                self.graph.entity_count(tenant_id)
            ),
        )

    def get_platform_stats(self) -> dict:
        """Get platform-wide statistics."""
        tenants = self.tenants.list_tenants(active_only=False)
        tenant_stats = [self.get_tenant_stats(t.tenant_id) for t in tenants]

        return {
            "total_tenants": len(tenants),
            "active_tenants": sum(1 for t in tenants if t.is_active),
            "total_memories": sum(ts.total_memories for ts in tenant_stats),
            "total_rules": sum(ts.procedural_count for ts in tenant_stats),
            "total_graph_entities": sum(ts.graph_entities for ts in tenant_stats),
            "tenants": [ts.model_dump() for ts in tenant_stats],
        }
