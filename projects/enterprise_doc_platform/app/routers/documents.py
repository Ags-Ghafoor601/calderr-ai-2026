"""Document upload and management API endpoints."""

import asyncio
import os
import shutil

import aiofiles
from fastapi import APIRouter, HTTPException, UploadFile, File, BackgroundTasks

from app.config import settings
from app.models import DocumentResponse, DocumentListResponse
from app.services.document_processor import (
    register_document, get_document, list_tenant_documents,
    get_tenant, process_document,
)
from app.services.rag_engine import rag_engine

router = APIRouter(prefix="/documents", tags=["Documents"])


@router.post("/{tenant_id}/upload", response_model=DocumentResponse, status_code=202)
async def upload_document(
    tenant_id: str,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    """Upload a document for processing. Returns immediately with a document ID.

    The document is processed asynchronously in the background.
    Check status via GET /documents/{tenant_id}/{document_id}.
    """
    # Validate tenant exists
    if not get_tenant(tenant_id):
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found")

    # Validate file type
    allowed_extensions = {".pdf", ".txt", ".md", ".markdown"}
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Allowed: {allowed_extensions}"
        )

    # Save uploaded file
    tenant_upload_dir = os.path.join(settings.upload_dir, tenant_id)
    os.makedirs(tenant_upload_dir, exist_ok=True)
    file_path = os.path.join(tenant_upload_dir, file.filename)

    async with aiofiles.open(file_path, "wb") as f:
        content = await file.read()
        await f.write(content)

    file_size = len(content)

    # Register document
    document_id = register_document(tenant_id, file.filename, file_size)

    # Process in background
    background_tasks.add_task(
        _run_async_processing,
        tenant_id, document_id, file_path, file.filename,
    )

    doc = get_document(document_id)
    return DocumentResponse(**doc)


def _run_async_processing(tenant_id, document_id, file_path, filename):
    """Wrapper to run async processing in a sync background task."""
    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(
            process_document(tenant_id, document_id, file_path, filename)
        )
        # Invalidate BM25 cache after new document
        rag_engine.invalidate_bm25(tenant_id)
    finally:
        loop.close()


@router.get("/{tenant_id}", response_model=DocumentListResponse)
async def list_documents(tenant_id: str):
    """List all documents for a tenant."""
    if not get_tenant(tenant_id):
        raise HTTPException(status_code=404, detail=f"Tenant '{tenant_id}' not found")

    docs = list_tenant_documents(tenant_id)
    return DocumentListResponse(
        tenant_id=tenant_id,
        documents=[DocumentResponse(**d) for d in docs],
        total=len(docs),
    )


@router.get("/{tenant_id}/{document_id}", response_model=DocumentResponse)
async def get_document_status(tenant_id: str, document_id: str):
    """Get the processing status of a document."""
    doc = get_document(document_id)
    if not doc or doc.get("tenant_id") != tenant_id:
        raise HTTPException(status_code=404, detail="Document not found")

    return DocumentResponse(**doc)
