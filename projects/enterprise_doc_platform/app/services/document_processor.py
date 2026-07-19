"""Enterprise Document Intelligence Platform — Document Processing Pipeline.

Handles document parsing, chunking, and async ingestion into the vector store.
Supports PDF, TXT, and Markdown formats.
"""

import asyncio
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path

from app.config import settings
from app.models import DocumentStatus

logger = logging.getLogger(__name__)

# In-memory document registry (in production, use a real database)
_document_registry: dict[str, dict] = {}
_tenant_registry: dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Tenant Management
# ---------------------------------------------------------------------------
def create_tenant(name: str, description: str = "") -> dict:
    """Register a new tenant."""
    tenant_id = name.lower().replace(" ", "_").replace("-", "_")[:30]
    if tenant_id in _tenant_registry:
        return _tenant_registry[tenant_id]

    tenant = {
        "tenant_id": tenant_id,
        "name": name,
        "description": description,
        "created_at": datetime.now().isoformat(),
    }
    _tenant_registry[tenant_id] = tenant
    logger.info("Created tenant: %s (%s)", tenant_id, name)
    return tenant


def get_tenant(tenant_id: str) -> dict | None:
    """Get tenant info."""
    return _tenant_registry.get(tenant_id)


def list_tenants() -> list[dict]:
    """List all registered tenants."""
    return list(_tenant_registry.values())


def delete_tenant(tenant_id: str) -> bool:
    """Delete a tenant and all their data."""
    if tenant_id in _tenant_registry:
        del _tenant_registry[tenant_id]
        # Remove associated documents
        doc_ids_to_remove = [
            did for did, doc in _document_registry.items()
            if doc.get("tenant_id") == tenant_id
        ]
        for did in doc_ids_to_remove:
            del _document_registry[did]
        return True
    return False


# ---------------------------------------------------------------------------
# Document Parsing
# ---------------------------------------------------------------------------
def parse_document(file_path: str, filename: str) -> list[dict]:
    """Parse a document file and extract pages/sections.

    Returns list of {"text": str, "metadata": dict}.
    """
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _parse_pdf(file_path, filename)
    elif suffix in (".txt", ".md", ".markdown"):
        return _parse_text(file_path, filename)
    else:
        raise ValueError(f"Unsupported file format: {suffix}")


def _parse_pdf(file_path: str, filename: str) -> list[dict]:
    """Parse a PDF file into pages."""
    import fitz

    documents = []
    doc = fitz.open(file_path)

    for page_num, page in enumerate(doc, 1):
        text = page.get_text().strip()
        if text and len(text) > 30:
            documents.append({
                "text": text,
                "metadata": {
                    "source": filename,
                    "page": page_num,
                    "total_pages": len(doc),
                    "format": "pdf",
                },
            })

    doc.close()
    return documents


def _parse_text(file_path: str, filename: str) -> list[dict]:
    """Parse a text/markdown file."""
    text = Path(file_path).read_text(encoding="utf-8", errors="ignore").strip()
    if not text or len(text) < 30:
        return []

    # Split by headers for markdown
    sections = []
    if filename.endswith((".md", ".markdown")):
        current_section = ""
        for line in text.split("\n"):
            if line.startswith("# ") and current_section.strip():
                sections.append(current_section.strip())
                current_section = line + "\n"
            else:
                current_section += line + "\n"
        if current_section.strip():
            sections.append(current_section.strip())
    else:
        sections = [text]

    documents = []
    for i, section in enumerate(sections, 1):
        if len(section) > 30:
            documents.append({
                "text": section,
                "metadata": {
                    "source": filename,
                    "page": i,
                    "total_pages": len(sections),
                    "format": Path(filename).suffix.lstrip("."),
                },
            })

    return documents


# ---------------------------------------------------------------------------
# Text Chunking
# ---------------------------------------------------------------------------
def chunk_documents(
    documents: list[dict],
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> list[dict]:
    """Split documents into chunks for embedding."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    cs = chunk_size or settings.chunk_size
    co = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cs,
        chunk_overlap=co,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["text"])
        for i, text in enumerate(splits):
            chunks.append({
                "text": text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": i,
                    "chunk_total": len(splits),
                },
            })

    return chunks


# ---------------------------------------------------------------------------
# Async Processing Pipeline
# ---------------------------------------------------------------------------
async def process_document(
    tenant_id: str,
    document_id: str,
    file_path: str,
    filename: str,
):
    """Process a document asynchronously: parse → chunk → embed → store."""
    from app.services.vector_store import vector_store

    # Update status
    if document_id in _document_registry:
        _document_registry[document_id]["status"] = DocumentStatus.PROCESSING

    try:
        # 1. Parse document
        logger.info("Parsing document: %s", filename)
        documents = await asyncio.to_thread(parse_document, file_path, filename)

        if not documents:
            raise ValueError("No content extracted from document")

        # 2. Chunk
        logger.info("Chunking %d sections...", len(documents))
        chunks = await asyncio.to_thread(chunk_documents, documents)

        # 3. Store in vector DB
        logger.info("Storing %d chunks for tenant '%s'...", len(chunks), tenant_id)
        num_stored = await asyncio.to_thread(
            vector_store.add_chunks, tenant_id, chunks, document_id
        )

        # 4. Update registry
        if document_id in _document_registry:
            _document_registry[document_id].update({
                "status": DocumentStatus.READY,
                "chunk_count": num_stored,
                "processed_at": datetime.now().isoformat(),
            })

        logger.info(
            "Document '%s' processed: %d chunks stored for tenant '%s'",
            filename, num_stored, tenant_id,
        )

    except Exception as e:
        logger.error("Failed to process document '%s': %s", filename, e)
        if document_id in _document_registry:
            _document_registry[document_id]["status"] = DocumentStatus.FAILED


def register_document(tenant_id: str, filename: str, file_size: int) -> str:
    """Register a new document and return its ID."""
    document_id = f"{tenant_id}_{uuid.uuid4().hex[:8]}"
    _document_registry[document_id] = {
        "document_id": document_id,
        "tenant_id": tenant_id,
        "filename": filename,
        "file_size": file_size,
        "status": DocumentStatus.PENDING,
        "chunk_count": 0,
        "uploaded_at": datetime.now().isoformat(),
        "processed_at": None,
    }
    return document_id


def get_document(document_id: str) -> dict | None:
    """Get document info."""
    return _document_registry.get(document_id)


def list_tenant_documents(tenant_id: str) -> list[dict]:
    """List all documents for a tenant."""
    return [
        doc for doc in _document_registry.values()
        if doc.get("tenant_id") == tenant_id
    ]
