"""
CalderR Internship – Week 5, Project 5-P-A
=============================================
Autonomous AI Research Lab — FastAPI REST API

Endpoints:
  GET  /health              — Health check with system info
  POST /research            — Run full 5-phase research pipeline
  GET  /domains             — List supported research domains
  GET  /reports             — List saved reports
  GET  /reports/{filename}  — Get a specific saved report

Run:
    uvicorn projects.research_lab.api:api --reload --port 8001
"""

import os
import sys
import json
import time
from pathlib import Path

# Fix imports
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

from projects.research_lab.models import (
    HealthResponse, ResearchRequest, ResearchResponse, ResearchDomain,
)
from projects.research_lab.domain_classifier import DomainClassifier

PROJECT_DIR = Path(__file__).resolve().parent
REPORTS_DIR = PROJECT_DIR / "reports"

api = FastAPI(
    title="Autonomous AI Research Lab API",
    description=(
        "Multi-agent research system with 5-phase pipeline: "
        "Hypothesis → Evidence → Critique → Synthesis → Peer Review. "
        "Dynamic agent assembly based on domain classification."
    ),
    version="1.0.0",
)

api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@api.get("/health", response_model=HealthResponse)
async def health():
    """Health check with system information."""
    return HealthResponse(
        status="healthy",
        project="Autonomous AI Research Lab",
        version="1.0.0",
        phases=5,
    )


@api.get("/domains")
async def list_domains():
    """List all supported research domains and their agent teams."""
    classifier = DomainClassifier()
    domains = []
    for d in ResearchDomain:
        team = classifier.assemble_team(d)
        domains.append({
            "domain": d.value,
            "agents": [a["name"] for a in team],
            "agent_count": len(team),
        })
    return {"domains": domains}


@api.post("/research", response_model=ResearchResponse)
async def run_research(request: ResearchRequest):
    """Run the full 5-phase research pipeline."""
    start = time.time()
    topic = request.topic.strip()

    if not topic or len(topic) < 5:
        raise HTTPException(status_code=400, detail="Topic must be at least 5 characters")

    try:
        from projects.research_lab.main import run_research_pipeline

        report = run_research_pipeline(
            topic,
            domain_override=request.domain,
            verbose=False,
        )

        elapsed = (time.time() - start) * 1000

        # Save report
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = topic[:30].lower().replace(" ", "_").replace("/", "_")
        out_path = REPORTS_DIR / f"report_{safe_name}.json"
        out_path.write_text(
            json.dumps(report.model_dump(), indent=2, default=str),
            encoding="utf-8",
        )

        return ResearchResponse(
            status="success",
            topic=topic,
            domain=report.domain.value,
            report=report.model_dump(),
            processing_time_ms=round(elapsed, 1),
        )

    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return ResearchResponse(
            status="error",
            topic=topic,
            error=str(e)[:500],
            processing_time_ms=round(elapsed, 1),
        )


@api.get("/reports")
async def list_reports():
    """List all saved research reports."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reports = []
    for f in REPORTS_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            reports.append({
                "filename": f.name,
                "topic": data.get("topic", "Unknown"),
                "domain": data.get("domain", "Unknown"),
                "quality_score": data.get("overall_quality_score", 0),
                "size_bytes": f.stat().st_size,
            })
        except Exception:
            reports.append({"filename": f.name, "size_bytes": f.stat().st_size})
    return {"reports": reports, "count": len(reports)}


@api.get("/reports/{filename}")
async def get_report(filename: str):
    """Get a specific saved report by filename."""
    report_path = REPORTS_DIR / filename
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Report '{filename}' not found")
    return json.loads(report_path.read_text(encoding="utf-8"))
