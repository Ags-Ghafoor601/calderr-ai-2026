# Enterprise AI Memory Platform

> **CalderR Internship — Week 6, Production Project (6-P-A)**
> A production-grade memory-as-a-service platform for AI agents.

## Overview

This project builds a **complete memory platform** that any AI agent can connect to via REST API. It provides all four memory types — episodic, semantic, procedural, and knowledge graph — with multi-tenant isolation, cross-session persistence, and a rich admin dashboard.

**This is what Mem0 raised $10M to build.** Our open-source version demonstrates product-level engineering with full API documentation, tenant isolation verification, and a working agent integration.

## Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                ENTERPRISE AI MEMORY PLATFORM                             │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │                     FastAPI Memory Service                       │     │
│  │                     (REST API, port 8000)                        │     │
│  │                                                                  │     │
│  │  /memory/{tenant}/episodic   → Store & query interaction events  │     │
│  │  /memory/{tenant}/semantic   → Store & search facts/preferences  │     │
│  │  /memory/{tenant}/procedural → Store & apply correction rules    │     │
│  │  /memory/{tenant}/graph/*    → Entity-relationship CRUD + paths  │     │
│  │  /consolidation/*            → Trigger & manage consolidation    │     │
│  │  /verify/isolation           → Multi-tenant isolation test       │     │
│  └──────────────┬───────────────────────────────────────────────────┘     │
│                 │                                                        │
│                 ▼                                                        │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │                      Memory Router                               │     │
│  │                                                                  │     │
│  │  Routes requests to the correct store based on memory type.      │     │
│  │  Handles automatic tenant provisioning.                          │     │
│  │  Provides cross-store statistics.                                │     │
│  └───┬──────────┬──────────┬──────────┬─────────────────────────────┘     │
│      │          │          │          │                                   │
│      ▼          ▼          ▼          ▼                                   │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────────┐                         │
│  │Episodic│ │Semantic│ │Procedur│ │Knowledge   │                         │
│  │Store   │ │Store   │ │Store   │ │Graph Store │                         │
│  │────────│ │────────│ │────────│ │────────────│                         │
│  │SQLite  │ │ChromaDB│ │SQLite  │ │NetworkX +  │                         │
│  │per-    │ │per-    │ │per-    │ │JSON per-   │                         │
│  │tenant  │ │tenant  │ │tenant  │ │tenant      │                         │
│  │tables  │ │collect.│ │tables  │ │             │                         │
│  └────────┘ └────────┘ └────────┘ └────────────┘                         │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │                  Consolidation Worker                            │     │
│  │                                                                  │     │
│  │  Summarises old episodes → semantic memory                       │     │
│  │  Promotes high-usage procedural rules                            │     │
│  │  Prunes low-importance consolidated memories                     │     │
│  │  Merges similar procedural rules                                 │     │
│  └──────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │                  Streamlit Admin Dashboard                       │     │
│  │                  (port 8501)                                     │     │
│  │                                                                  │     │
│  │  Platform Overview • Memory Inspector • Knowledge Graph Viz      │     │
│  │  Consolidation Manager • Isolation Checker • Demo Data Seeder    │     │
│  └──────────────────────────────────────────────────────────────────┘     │
│                                                                          │
│  ┌──────────────────────────────────────────────────────────────────┐     │
│  │           External Agent Integration (via REST API)              │     │
│  │                                                                  │     │
│  │  MemoryClient → HTTP → FastAPI → Memory Router → Stores          │     │
│  │  Any LangChain / CrewAI / custom agent can connect               │     │
│  └──────────────────────────────────────────────────────────────────┘     │
└──────────────────────────────────────────────────────────────────────────┘
```

## Memory Types

### 1. Episodic Memory (SQLite)
Timestamped interaction events with importance scoring:

| Column | Type | Purpose |
|--------|------|---------|
| `memory_id` | TEXT PK | Unique identifier |
| `tenant_id` | TEXT | Multi-tenant isolation |
| `session_id` | TEXT | Group interactions into sessions |
| `timestamp` | TEXT | ISO 8601 UTC |
| `content` | TEXT | The interaction content |
| `role` | TEXT | user / assistant |
| `importance_score` | REAL | 0.0–1.0, for consolidation decisions |
| `is_consolidated` | INT | 0=raw, 1=compressed into summary |

### 2. Semantic Memory (ChromaDB)
Embedded facts and preferences with similarity search:
- Per-tenant namespaced collections (`semantic_{tenant_id}`)
- Category filtering (profile, preference, fact, knowledge)
- Cosine similarity search with relevance scoring

### 3. Procedural Memory (SQLite)
Correction rules that prevent repeated mistakes:
- Domain classification (factual, formatting, tone, reasoning, accuracy, completeness)
- Confidence scoring with automatic promotion on usage
- Rule consolidation (3+ similar rules → merged into one)

### 4. Knowledge Graph (NetworkX + JSON)
Entity-relationship graph per tenant:
- Typed entities (person, company, concept, place, product)
- Typed relationships (founded_by, ceo_of, competes_with, etc.)
- Multi-hop traversal and shortest path finding
- JSON persistence per tenant

## Project Structure

```
enterprise_memory_platform/
├── api.py                      # FastAPI service (main entry point)
├── dashboard.py                # Streamlit admin dashboard
├── router.py                   # Central memory router
├── consolidation.py            # Background consolidation worker
├── models.py                   # Pydantic schemas for all memory types
├── integration_example.py      # LangChain agent integration demo
├── stores/
│   ├── __init__.py
│   ├── episodic.py             # SQLite episodic store
│   ├── semantic.py             # ChromaDB semantic store
│   ├── procedural.py           # SQLite procedural store
│   ├── knowledge_graph.py      # NetworkX graph store
│   └── tenants.py              # Tenant manager
├── Dockerfile                  # API container
├── docker-compose.yml          # One-command deployment
├── requirements.txt            # Python dependencies
├── architecture.png            # Architecture diagram
└── README.md                   # This file
```

## Quick Start

### Local Development

```bash
# Install dependencies
cd projects/enterprise_memory_platform
pip install -r requirements.txt

# Start the FastAPI service
uvicorn api:app --reload --port 8000
# → API docs at http://localhost:8000/docs

# Start the admin dashboard (in another terminal)
streamlit run dashboard.py
# → Dashboard at http://localhost:8501

# Run the integration example (requires API running)
python integration_example.py
```

### Docker Compose (One-Command)

```bash
docker-compose up --build
# → API at http://localhost:8000
# → Dashboard at http://localhost:8501
```

## API Documentation

Full OpenAPI documentation is auto-generated at `/docs` (Swagger UI) and `/redoc` (ReDoc).

### Key Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Health check |
| `GET` | `/stats` | Platform-wide statistics |
| `GET` | `/tenants` | List all tenants |
| `POST` | `/tenants` | Create a new tenant |
| `GET` | `/tenants/{id}/stats` | Tenant-specific statistics |
| | | |
| `POST` | `/memory/{id}/episodic` | Store an episodic memory |
| `POST` | `/memory/{id}/episodic/query` | Query episodic memories |
| `DELETE` | `/memory/{id}/episodic/{mid}` | Delete an episodic memory |
| | | |
| `POST` | `/memory/{id}/semantic` | Store a semantic fact |
| `POST` | `/memory/{id}/semantic/query` | Search semantic memories |
| `GET` | `/memory/{id}/semantic/all` | List all semantic memories |
| | | |
| `POST` | `/memory/{id}/procedural` | Store a correction rule |
| `POST` | `/memory/{id}/procedural/query` | Query procedural rules |
| `POST` | `/memory/{id}/procedural/{rid}/apply` | Record rule application |
| | | |
| `POST` | `/memory/{id}/graph/entity` | Add a graph entity |
| `POST` | `/memory/{id}/graph/relationship` | Add a graph relationship |
| `POST` | `/memory/{id}/graph/query` | Query the graph |
| `GET` | `/memory/{id}/graph/path?source=A&target=B` | Find shortest path |
| | | |
| `POST` | `/consolidation/{id}` | Trigger consolidation |
| `POST` | `/consolidation/run-all` | Consolidate all tenants |
| `GET` | `/verify/isolation` | Multi-tenant isolation test |

## Multi-Tenant Isolation

Every memory operation is scoped by `tenant_id`:
- **SQLite**: All queries include `WHERE tenant_id = ?`
- **ChromaDB**: Each tenant has a separate collection (`semantic_{tenant_id}`)
- **Knowledge Graph**: Each tenant has a separate JSON file and in-memory graph

The `/verify/isolation` endpoint seeds data for Acme Corp and then queries as Globex Inc — proving cross-tenant data is invisible.

## Consolidation Worker

The consolidation engine runs automatically or on-demand:

1. **Episode compression**: Old episodes (>100 per tenant) are summarised by LLM and stored as semantic memory
2. **Rule promotion**: Rules applied 3+ times get a confidence boost
3. **Rule consolidation**: 3+ similar rules are merged into one high-confidence rule
4. **Memory pruning**: Low-importance consolidated memories are cleaned up

## Evaluation Criteria

| Criterion | Status |
|-----------|--------|
| All 4 memory types accessible via REST API | ✅ |
| Multi-tenant isolation verified | ✅ |
| Consolidation worker runs correctly | ✅ |
| Admin dashboard shows accurate state | ✅ |
| Docker Compose starts full platform | ✅ |
| External agent integration example | ✅ |
| OpenAPI docs complete | ✅ |
| 3 demo tenants with realistic data | ✅ |

## What Makes This Stand Out

This is a **real product**. Mem0 raised $10M building exactly this. Our open-source version:
- Is deployed with one command (`docker-compose up`)
- Has complete OpenAPI documentation
- Includes a working agent integration example
- Features a rich admin dashboard for live memory inspection
- Verifies multi-tenant isolation programmatically
- Implements all 4 memory types with production-grade persistence
