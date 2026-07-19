"""Enterprise Document Intelligence Platform — FastAPI Application.

Multi-tenant document intelligence API with:
  • Document upload and async processing
  • Advanced RAG (hybrid search + cross-encoder re-ranking)
  • Tenant isolation via ChromaDB namespaces
  • Confidence-scored answers with source citations

API Documentation: http://localhost:8000/docs
"""

import time
import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.models import HealthResponse
from app.routers import tenants, documents, query

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------
START_TIME = time.time()

app = FastAPI(
    title=settings.app_name,
    version=settings.app_version,
    description="""
## Enterprise Document Intelligence Platform

Multi-tenant document intelligence API that processes documents, indexes them
using embeddings, and answers questions with confidence scores and source citations.

### Key Features
- **Multi-tenancy**: Each tenant's documents are isolated in separate ChromaDB namespaces
- **Async Processing**: Documents are processed in the background after upload
- **Hybrid RAG**: Combines BM25 keyword search with semantic vector search
- **Cross-encoder Re-ranking**: Improves result quality using neural re-ranking
- **Confidence Scoring**: Each answer includes a confidence score (0–1)
- **Source Citations**: Answers include source document and page references

### Architecture
```
Upload → Parse → Chunk → Embed → ChromaDB (per-tenant)
                                        ↓
Query → Hybrid Search → Re-rank → LLM → Cited Answer
```
    """,
    docs_url="/docs",
    redoc_url="/redoc",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(tenants.router, prefix="/api/v1")
app.include_router(documents.router, prefix="/api/v1")
app.include_router(query.router, prefix="/api/v1")


# ---------------------------------------------------------------------------
# Root & Health
# ---------------------------------------------------------------------------
@app.get("/", tags=["Root"])
async def root():
    """Platform welcome endpoint."""
    return {
        "name": settings.app_name,
        "version": settings.app_version,
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        version=settings.app_version,
        uptime_seconds=round(time.time() - START_TIME, 2),
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.debug,
    )
