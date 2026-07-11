"""
CalderR Internship – Week 2, Production Project
===================================================
Intelligent Document Processing Pipeline — FastAPI Application

WHAT THIS PROJECT BUILDS:
-------------------------
A FastAPI-powered REST API that:
  • Accepts document uploads (PDF, DOCX, TXT)
  • Parses document content using PyMuPDF / python-docx
  • Extracts structured data (entities, dates, key terms,
    summaries, action items) via Groq LLM
  • Validates everything with Pydantic v2
  • Stores results in SQLite
  • Serves a simple frontend for uploading and viewing results

ARCHITECTURE:
  File Upload (FastAPI)
      ↓
  Document Parser (PyMuPDF / python-docx / text)
      ↓
  Multi-Tool Extraction Agent (Groq + LangChain)
      ↓
  Pydantic Validation
      ↓
  Database (SQLite via aiosqlite)
      ↓
  REST API + Simple Frontend

Run:
    cd projects/doc_pipeline
    uvicorn main:app --reload --port 8000
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Load .env from repo root
_repo_root = Path(__file__).parent.parent.parent
load_dotenv(_repo_root / ".env")

# Local imports
from models import DocumentType, DocumentRecord, DocumentExtraction
from parsers.pdf_parser import parse_pdf
from parsers.docx_parser import parse_docx
from parsers.txt_parser import parse_txt
from extraction.extractor import extract_document
from database import init_db, save_document, get_document, get_all_documents, get_document_count, delete_document


# ═══════════════════════════════════════════════
#  App Setup
# ═══════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize database on startup."""
    await init_db()
    yield


app = FastAPI(
    title="Intelligent Document Processing Pipeline",
    description=(
        "Upload documents (PDF, DOCX, TXT) and extract structured information: "
        "entities, dates, key terms, summaries, and action items — all validated "
        "with Pydantic v2 and powered by Groq LLM."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# File type mapping
PARSERS = {
    "application/pdf": ("pdf", parse_pdf),
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ("docx", parse_docx),
    "text/plain": ("txt", parse_txt),
}

# Also match by extension
EXTENSION_MAP = {
    ".pdf": ("pdf", parse_pdf),
    ".docx": ("docx", parse_docx),
    ".txt": ("txt", parse_txt),
}

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB


# ═══════════════════════════════════════════════
#  API Endpoints
# ═══════════════════════════════════════════════

@app.post("/api/upload", response_model=dict, tags=["Documents"])
async def upload_document(file: UploadFile = File(...)):
    """
    Upload a document for processing.

    Accepts PDF, DOCX, or TXT files up to 10MB.
    Returns structured extraction results including entities,
    dates, key terms, summary, and action items.
    """
    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided")

    # Determine file type from extension
    ext = Path(file.filename).suffix.lower()
    if ext not in EXTENSION_MAP:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: '{ext}'. Supported: .pdf, .docx, .txt",
        )

    file_type_str, parser_func = EXTENSION_MAP[ext]

    # Read file content
    content = await file.read()
    file_size = len(content)

    if file_size == 0:
        raise HTTPException(status_code=400, detail="File is empty")
    if file_size > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=400,
            detail=f"File too large ({file_size / 1024 / 1024:.1f}MB). Max: 10MB",
        )

    start_time = time.time()

    # Step 1: Parse document
    try:
        text = parser_func(content)
    except ValueError as e:
        raise HTTPException(status_code=422, detail=f"Parsing error: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected parsing error: {str(e)}")

    # Step 2: Extract structured data via LLM
    try:
        extraction = extract_document(text)
    except Exception as e:
        # Save a failed record
        error_record = DocumentRecord(
            filename=file.filename,
            file_type=DocumentType(file_type_str),
            file_size_bytes=file_size,
            extraction=DocumentExtraction(
                summary=f"Extraction failed: {str(e)}",
            ),
            processing_time_ms=(time.time() - start_time) * 1000,
            status="failed",
            error=str(e),
        )
        doc_id = await save_document(error_record.model_dump(mode="json"))
        raise HTTPException(
            status_code=500,
            detail=f"Extraction failed: {str(e)}. Record saved with ID {doc_id}.",
        )

    processing_time = (time.time() - start_time) * 1000

    # Step 3: Save to database
    record = DocumentRecord(
        filename=file.filename,
        file_type=DocumentType(file_type_str),
        file_size_bytes=file_size,
        extraction=extraction,
        processing_time_ms=processing_time,
        status="completed",
    )

    doc_id = await save_document(record.model_dump(mode="json"))

    return {
        "id": doc_id,
        "filename": file.filename,
        "file_type": file_type_str,
        "file_size_bytes": file_size,
        "processing_time_ms": round(processing_time, 1),
        "status": "completed",
        "extraction": extraction.model_dump(mode="json"),
    }


@app.get("/api/documents", tags=["Documents"])
async def list_documents(limit: int = 50, offset: int = 0):
    """
    List all processed documents with pagination.
    """
    documents = await get_all_documents(limit=limit, offset=offset)
    total = await get_document_count()

    return {
        "documents": documents,
        "total": total,
        "limit": limit,
        "offset": offset,
    }


@app.get("/api/documents/{doc_id}", tags=["Documents"])
async def get_document_by_id(doc_id: int):
    """
    Get a specific document's extraction results by ID.
    """
    doc = await get_document(doc_id)
    if doc is None:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return doc


@app.delete("/api/documents/{doc_id}", tags=["Documents"])
async def delete_document_by_id(doc_id: int):
    """
    Delete a document record by ID.
    """
    deleted = await delete_document(doc_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Document {doc_id} not found")
    return {"message": f"Document {doc_id} deleted successfully"}


@app.get("/api/health", tags=["System"])
async def health_check():
    """Health check endpoint."""
    doc_count = await get_document_count()
    return {
        "status": "healthy",
        "version": "1.0.0",
        "documents_processed": doc_count,
        "timestamp": datetime.now().isoformat(),
    }


# ═══════════════════════════════════════════════
#  Frontend
# ═══════════════════════════════════════════════

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def frontend():
    """Serve the frontend HTML page."""
    frontend_path = Path(__file__).parent / "frontend" / "index.html"
    if frontend_path.exists():
        return HTMLResponse(content=frontend_path.read_text(encoding="utf-8"))
    return HTMLResponse(content="<h1>Frontend not found</h1>", status_code=404)
