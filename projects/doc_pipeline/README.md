# Intelligent Document Processing Pipeline

## 📋 Overview
A production-grade FastAPI application that processes uploaded documents (PDF, DOCX, TXT) and extracts structured information — entities, dates, key terms, summaries, and action items — all validated with Pydantic v2 and powered by Groq LLM.

## 🏗️ Architecture
```
File Upload (FastAPI)
    ↓
Document Parser (PyMuPDF / python-docx / text)
    ↓
Multi-Tool Extraction Agent (Groq + LangChain)
    ↓
Pydantic Validation (DocumentExtraction model)
    ↓
Database (SQLite via aiosqlite)
    ↓
REST API + Frontend UI
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Groq API key in `.env` file at repo root

### Local Development
```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
cd projects/doc_pipeline
uvicorn main:app --reload --port 8000

# Open in browser
# http://localhost:8000        — Frontend UI
# http://localhost:8000/docs   — Swagger API docs
# http://localhost:8000/redoc  — ReDoc API docs
```

### Docker Deployment
```bash
cd projects/doc_pipeline

# Set your API key
export GROQ_API_KEY=your_key_here

# Build and run
docker compose up --build

# Or just build
docker compose build
docker compose up -d
```

## 🔌 API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload a document for processing |
| `GET` | `/api/documents` | List all processed documents |
| `GET` | `/api/documents/{id}` | Get specific document results |
| `DELETE` | `/api/documents/{id}` | Delete a document record |
| `GET` | `/api/health` | Health check |
| `GET` | `/` | Frontend UI |

### Upload Example (cURL)
```bash
curl -X POST http://localhost:8000/api/upload \
  -F "file=@document.pdf"
```

### Response Format
```json
{
    "id": 1,
    "filename": "report.pdf",
    "file_type": "pdf",
    "processing_time_ms": 2340.5,
    "status": "completed",
    "extraction": {
        "summary": "This document discusses...",
        "key_terms": ["AI", "machine learning", "deployment"],
        "entities": [
            {"text": "Google", "entity_type": "organization", "confidence": 0.95}
        ],
        "dates": [
            {"text": "January 2026", "normalized": "2026-01-01", "context": "Project deadline"}
        ],
        "action_items": [
            {"description": "Submit final report", "severity": "high", "deadline": "Friday"}
        ],
        "document_type_guess": "report",
        "language": "English",
        "word_count": 1250
    }
}
```

## 📁 Project Structure
```
projects/doc_pipeline/
├── main.py                 # FastAPI application entry point
├── models.py               # Pydantic v2 models (entities, dates, actions)
├── database.py             # SQLite async database operations
├── parsers/
│   ├── __init__.py
│   ├── pdf_parser.py       # PyMuPDF-based PDF text extraction
│   ├── docx_parser.py      # python-docx based DOCX extraction
│   └── txt_parser.py       # Plain text with multi-encoding support
├── extraction/
│   ├── __init__.py
│   └── extractor.py        # Groq LLM extraction agent
├── frontend/
│   └── index.html          # Dark-themed upload + results UI
├── test_documents/         # Sample documents for testing
├── requirements.txt        # Project dependencies
├── Dockerfile              # Docker image definition
├── docker-compose.yml      # Docker Compose config
└── README.md               # This file
```

## 📊 Extracted Information

| Category | Details |
|----------|---------|
| **Summary** | 2-3 sentence overview of the document |
| **Key Terms** | 5-15 important concepts and keywords |
| **Entities** | People, organizations, locations, dates, money, emails, URLs |
| **Dates** | All dates and deadlines with ISO normalization |
| **Action Items** | Tasks with severity levels (high/medium/low) |
| **Document Type** | Auto-classified (contract, report, email, memo, etc.) |

## 🔑 Key Features
- **Multi-format support**: PDF, DOCX, TXT
- **AI-powered extraction**: Groq LLM with structured output
- **Type-safe validation**: Pydantic v2 models enforce output schema
- **Async database**: aiosqlite for non-blocking SQLite operations
- **Beautiful frontend**: Dark-themed UI with drag-and-drop upload
- **Auto-generated docs**: Swagger + ReDoc API documentation
- **Docker-ready**: Dockerfile + Docker Compose for easy deployment
- **Robust error handling**: Graceful failures with error records

## 📊 Skills Demonstrated
- FastAPI application design with async endpoints
- File processing (PDF, DOCX, TXT parsing)
- Structured extraction with LangChain + Groq
- Pydantic v2 advanced validation (field_validator, model_validator)
- Async database operations with aiosqlite
- Docker containerization
- Frontend development with vanilla JS
- REST API design with proper error handling
