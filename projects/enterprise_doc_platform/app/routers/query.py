"""RAG Query API endpoint."""

from fastapi import APIRouter, HTTPException

from app.models import QueryRequest, QueryResponse, SourceChunk
from app.services.document_processor import get_tenant
from app.services.rag_engine import rag_engine

router = APIRouter(prefix="/query", tags=["Query"])


@router.post("/{tenant_id}", response_model=QueryResponse)
async def query_documents(tenant_id: str, request: QueryRequest):
    """Query a tenant's documents using advanced RAG.

    Uses hybrid search (BM25 + semantic) with optional cross-encoder re-ranking.
    Returns the answer with confidence scores, source chunks, and page numbers.
    """
    if not get_tenant(tenant_id):
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found")

    try:
        result = rag_engine.query(
            tenant_id=tenant_id,
            question=request.question,
            top_k=request.top_k,
            use_hybrid=request.use_hybrid,
            use_reranking=request.use_reranking,
        )

        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            confidence_score=result["confidence_score"],
            source_chunks=[SourceChunk(**sc) for sc in result["source_chunks"]],
            retrieval_method=result["retrieval_method"],
            processing_time_ms=result["processing_time_ms"],
            tenant_id=result["tenant_id"],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Query processing failed: {str(e)}"
        )
