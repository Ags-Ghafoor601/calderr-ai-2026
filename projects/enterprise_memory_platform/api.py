"""
Enterprise AI Memory Platform — FastAPI Memory Service
========================================================
Production-grade REST API with complete OpenAPI documentation
for all 4 memory types, multi-tenant isolation, and consolidation.

Run:
    uvicorn api:app --reload --port 8000
    # OR
    python api.py
"""

import os
import sys
import asyncio
from pathlib import Path
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import (
    MemoryResponse,
    EpisodicMemoryCreate, EpisodicQueryRequest,
    SemanticMemoryCreate, SemanticQueryRequest,
    ProceduralRuleCreate, ProceduralQueryRequest,
    EntityCreate, RelationshipCreate, GraphQueryRequest,
    ConsolidationConfig, RuleDomain,
)
from router import MemoryRouter
from consolidation import ConsolidationWorker


# ─── Data directory ───────────────────────────────────────────────────────
DATA_DIR = str(Path(__file__).resolve().parent / "data")
Path(DATA_DIR).mkdir(parents=True, exist_ok=True)


# ─── Lifespan: initialise router & worker ─────────────────────────────────
@asynccontextmanager
async def lifespan(application: FastAPI):
    """Application lifespan: init stores and seed demo tenants."""
    application.state.router = MemoryRouter(data_dir=DATA_DIR)
    application.state.worker = ConsolidationWorker(application.state.router)

    # Seed 3 demo tenants
    r = application.state.router
    r.tenants.create_tenant("acme_corp", "Acme Corporation")
    r.tenants.create_tenant("globex_inc", "Globex Inc.")
    r.tenants.create_tenant("initech", "Initech")

    yield


# ─── FastAPI App ──────────────────────────────────────────────────────────
app = FastAPI(
    title="Enterprise AI Memory Platform",
    description=(
        "A production-grade memory-as-a-service platform for AI agents.\n\n"
        "**4 Memory Types:**\n"
        "- **Episodic** — Timestamped interaction history with importance scoring\n"
        "- **Semantic** — Facts, preferences, and knowledge with similarity search\n"
        "- **Procedural** — Correction rules that prevent repeated mistakes\n"
        "- **Knowledge Graph** — Entity-relationship graph with multi-hop traversal\n\n"
        "**Multi-Tenant Isolation:** Each tenant has separate, isolated memory namespaces.\n\n"
        "**Consolidation:** Background worker summarises old episodes, promotes rules, prunes low-importance memories."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _get_router() -> MemoryRouter:
    return app.state.router


def _get_worker() -> ConsolidationWorker:
    return app.state.worker


# ═══════════════════════════════════════════════════════════════════════════
#  HEALTH & PLATFORM ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/", tags=["Platform"], summary="Platform health check")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "Enterprise AI Memory Platform", "version": "1.0.0"}


@app.get("/stats", tags=["Platform"], summary="Platform-wide statistics")
async def platform_stats():
    """Get platform-wide statistics across all tenants."""
    router = _get_router()
    return router.get_platform_stats()


@app.get("/tenants", tags=["Tenants"], summary="List all tenants")
async def list_tenants(active_only: bool = Query(True)):
    """List all registered tenants."""
    router = _get_router()
    tenants = router.tenants.list_tenants(active_only=active_only)
    return {"tenants": [t.model_dump() for t in tenants], "count": len(tenants)}


@app.post("/tenants", tags=["Tenants"], summary="Create a new tenant")
async def create_tenant(tenant_id: str = Query(...), name: str = Query(...)):
    """Create a new tenant with isolated memory namespaces."""
    router = _get_router()
    tenant = router.tenants.create_tenant(tenant_id, name)
    return {"success": True, "tenant": tenant.model_dump()}


@app.get("/tenants/{tenant_id}/stats", tags=["Tenants"], summary="Get tenant statistics")
async def tenant_stats(tenant_id: str):
    """Get comprehensive memory statistics for a tenant."""
    router = _get_router()
    if not router.tenants.tenant_exists(tenant_id):
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found")
    stats = router.get_tenant_stats(tenant_id)
    return stats.model_dump()


# ═══════════════════════════════════════════════════════════════════════════
#  EPISODIC MEMORY ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/memory/{tenant_id}/episodic", tags=["Episodic Memory"],
          summary="Store an episodic memory", response_model=MemoryResponse)
async def store_episodic(tenant_id: str, memory: EpisodicMemoryCreate):
    """Store a new episodic memory (interaction event) for a tenant."""
    router = _get_router()
    return router.store_episodic(tenant_id, memory)


@app.post("/memory/{tenant_id}/episodic/query", tags=["Episodic Memory"],
          summary="Query episodic memories", response_model=MemoryResponse)
async def query_episodic(tenant_id: str, request: EpisodicQueryRequest):
    """Query episodic memories by session, recency, or importance."""
    router = _get_router()
    return router.query_episodic(tenant_id, request)


@app.delete("/memory/{tenant_id}/episodic/{memory_id}", tags=["Episodic Memory"],
            summary="Delete an episodic memory", response_model=MemoryResponse)
async def delete_episodic(tenant_id: str, memory_id: str):
    """Delete a specific episodic memory."""
    router = _get_router()
    return router.delete_episodic(tenant_id, memory_id)


@app.get("/memory/{tenant_id}/episodic/sessions", tags=["Episodic Memory"],
         summary="List sessions")
async def list_sessions(tenant_id: str):
    """List all sessions for a tenant."""
    router = _get_router()
    sessions = router.episodic.get_sessions(tenant_id)
    return {"sessions": sessions, "count": len(sessions)}


# ═══════════════════════════════════════════════════════════════════════════
#  SEMANTIC MEMORY ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/memory/{tenant_id}/semantic", tags=["Semantic Memory"],
          summary="Store a semantic memory", response_model=MemoryResponse)
async def store_semantic(tenant_id: str, memory: SemanticMemoryCreate):
    """Store a semantic fact, preference, or piece of knowledge for a tenant."""
    router = _get_router()
    return router.store_semantic(tenant_id, memory)


@app.post("/memory/{tenant_id}/semantic/query", tags=["Semantic Memory"],
          summary="Query semantic memories", response_model=MemoryResponse)
async def query_semantic(tenant_id: str, request: SemanticQueryRequest):
    """Search semantic memories by similarity to a query string."""
    router = _get_router()
    return router.query_semantic(tenant_id, request)


@app.delete("/memory/{tenant_id}/semantic/{memory_id}", tags=["Semantic Memory"],
            summary="Delete a semantic memory", response_model=MemoryResponse)
async def delete_semantic(tenant_id: str, memory_id: str):
    """Delete a specific semantic memory."""
    router = _get_router()
    return router.delete_semantic(tenant_id, memory_id)


@app.get("/memory/{tenant_id}/semantic/all", tags=["Semantic Memory"],
         summary="List all semantic memories")
async def list_semantic(tenant_id: str, limit: int = Query(100, ge=1, le=500)):
    """Get all semantic memories for a tenant."""
    router = _get_router()
    memories = router.semantic.get_all(tenant_id, limit=limit)
    return {"memories": [m.model_dump() for m in memories], "count": len(memories)}


# ═══════════════════════════════════════════════════════════════════════════
#  PROCEDURAL MEMORY ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/memory/{tenant_id}/procedural", tags=["Procedural Memory"],
          summary="Store a procedural rule", response_model=MemoryResponse)
async def store_procedural(tenant_id: str, rule: ProceduralRuleCreate):
    """Store a new procedural correction rule for a tenant."""
    router = _get_router()
    return router.store_procedural(tenant_id, rule)


@app.post("/memory/{tenant_id}/procedural/query", tags=["Procedural Memory"],
          summary="Query procedural rules", response_model=MemoryResponse)
async def query_procedural(tenant_id: str, request: ProceduralQueryRequest):
    """Query procedural rules by domain, confidence, or semantic similarity."""
    router = _get_router()
    return router.query_procedural(tenant_id, request)


@app.post("/memory/{tenant_id}/procedural/{rule_id}/apply", tags=["Procedural Memory"],
          summary="Record rule application")
async def apply_rule(tenant_id: str, rule_id: str):
    """Record that a procedural rule was applied (increments application count)."""
    router = _get_router()
    return router.apply_rule(tenant_id, rule_id)


@app.delete("/memory/{tenant_id}/procedural/{rule_id}", tags=["Procedural Memory"],
            summary="Delete a procedural rule", response_model=MemoryResponse)
async def delete_procedural(tenant_id: str, rule_id: str):
    """Delete a specific procedural rule."""
    router = _get_router()
    return router.delete_procedural(tenant_id, rule_id)


# ═══════════════════════════════════════════════════════════════════════════
#  KNOWLEDGE GRAPH ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/memory/{tenant_id}/graph/entity", tags=["Knowledge Graph"],
          summary="Add an entity", response_model=MemoryResponse)
async def add_entity(tenant_id: str, entity: EntityCreate):
    """Add or update an entity in the tenant's knowledge graph."""
    router = _get_router()
    return router.add_entity(tenant_id, entity)


@app.post("/memory/{tenant_id}/graph/relationship", tags=["Knowledge Graph"],
          summary="Add a relationship", response_model=MemoryResponse)
async def add_relationship(tenant_id: str, rel: RelationshipCreate):
    """Add a relationship between two entities in the tenant's knowledge graph."""
    router = _get_router()
    return router.add_relationship(tenant_id, rel)


@app.post("/memory/{tenant_id}/graph/query", tags=["Knowledge Graph"],
          summary="Query the knowledge graph", response_model=MemoryResponse)
async def query_graph(tenant_id: str, request: GraphQueryRequest):
    """Query the knowledge graph by entity neighbourhood or natural language."""
    router = _get_router()
    return router.query_graph(tenant_id, request)


@app.get("/memory/{tenant_id}/graph/stats", tags=["Knowledge Graph"],
         summary="Get graph statistics")
async def graph_stats(tenant_id: str):
    """Get statistics for a tenant's knowledge graph."""
    router = _get_router()
    resp = router.get_graph_stats(tenant_id)
    return resp.data


@app.get("/memory/{tenant_id}/graph/entities", tags=["Knowledge Graph"],
         summary="List all entities")
async def list_entities(tenant_id: str):
    """List all entities in a tenant's knowledge graph."""
    router = _get_router()
    entities = router.graph.get_all_entities(tenant_id)
    return {"entities": [e.model_dump() for e in entities], "count": len(entities)}


@app.get("/memory/{tenant_id}/graph/relationships", tags=["Knowledge Graph"],
         summary="List all relationships")
async def list_relationships(tenant_id: str):
    """List all relationships in a tenant's knowledge graph."""
    router = _get_router()
    rels = router.graph.get_all_relationships(tenant_id)
    return {"relationships": [r.model_dump() for r in rels], "count": len(rels)}


@app.get("/memory/{tenant_id}/graph/path", tags=["Knowledge Graph"],
         summary="Find path between entities")
async def find_path(tenant_id: str, source: str = Query(...), target: str = Query(...)):
    """Find the shortest path between two entities in the knowledge graph."""
    router = _get_router()
    return router.find_path(tenant_id, source, target)


# ═══════════════════════════════════════════════════════════════════════════
#  CONSOLIDATION ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════

@app.post("/consolidation/{tenant_id}", tags=["Consolidation"],
          summary="Trigger consolidation for a tenant")
async def trigger_consolidation(tenant_id: str, background_tasks: BackgroundTasks):
    """Trigger memory consolidation for a specific tenant."""
    worker = _get_worker()
    router = _get_router()
    if not router.tenants.tenant_exists(tenant_id):
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found")
    record = worker.force_consolidation(tenant_id)
    return record.model_dump()


@app.post("/consolidation/run-all", tags=["Consolidation"],
          summary="Trigger consolidation for all tenants")
async def trigger_all_consolidation(background_tasks: BackgroundTasks):
    """Trigger memory consolidation for ALL active tenants."""
    worker = _get_worker()
    records = worker.run_all_tenants()
    return {"records": [r.model_dump() for r in records], "count": len(records)}


@app.get("/consolidation/history", tags=["Consolidation"],
         summary="Get consolidation history")
async def consolidation_history(tenant_id: Optional[str] = Query(None)):
    """Get the consolidation run history."""
    worker = _get_worker()
    records = worker.get_history(tenant_id)
    return {"records": [r.model_dump() for r in records], "count": len(records)}


# ═══════════════════════════════════════════════════════════════════════════
#  MULTI-TENANT ISOLATION VERIFICATION
# ═══════════════════════════════════════════════════════════════════════════

@app.get("/verify/isolation", tags=["Verification"],
         summary="Verify multi-tenant isolation")
async def verify_isolation():
    """Verify that tenant A cannot read tenant B's memories.
    Seeds test data and cross-checks isolation."""
    router = _get_router()

    # Seed test data for acme_corp
    from models import EpisodicMemoryCreate, SemanticMemoryCreate
    router.store_episodic("acme_corp", EpisodicMemoryCreate(
        session_id="test-isolation", content="Acme secret data: Project Alpha", importance_score=0.9,
    ))
    router.store_semantic("acme_corp", SemanticMemoryCreate(
        fact="Acme's revenue is $50M", category="financial",
    ))

    # Query as globex_inc — should find NOTHING
    ep_result = router.query_episodic("globex_inc", EpisodicQueryRequest(limit=100))
    sem_result = router.query_semantic("globex_inc", SemanticQueryRequest(query="Acme secret"))

    acme_found_by_globex = False
    for item in ep_result.data or []:
        if "Acme" in str(item):
            acme_found_by_globex = True
    for item in sem_result.data or []:
        if "Acme" in str(item.get("fact", "")):
            acme_found_by_globex = True

    return {
        "isolation_verified": not acme_found_by_globex,
        "detail": "Globex cannot access Acme's memories" if not acme_found_by_globex else "ISOLATION BREACH DETECTED",
        "acme_episodic_count": router.episodic.count("acme_corp"),
        "globex_episodic_found": len(ep_result.data) if ep_result.data else 0,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
