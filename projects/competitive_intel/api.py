"""
CalderR Internship – Week 5, Project 5-I-A
=============================================
Competitive Intelligence Agent — FastAPI REST API

Endpoints:
  GET  /health          — Health check
  POST /analyse         — Run intelligence pipeline for a company
  GET  /reports         — List saved reports
  GET  /reports/{name}  — Get a specific report

Run:
    uvicorn projects.competitive_intel.api:api --reload --port 8000
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
from pydantic import BaseModel

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

from projects.competitive_intel.models import HealthResponse, AnalysisResponse
from projects.competitive_intel.agents import (
    OrchestratorAgent, MarketAgent, ProductAgent, TechStackAgent,
    NewsAgent, SentimentAgent, ConflictResolverAgent, SynthesisAgent,
)

PROJECT_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = PROJECT_DIR / "sample_reports"

api = FastAPI(
    title="Competitive Intelligence Agent API",
    description=(
        "Multi-agent competitive intelligence system. "
        "Analyses companies using 5 specialist agents + conflict resolution + synthesis."
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


class AnalyseRequest(BaseModel):
    company: str


@api.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint."""
    return HealthResponse(status="healthy", agents=5, version="1.0.0")


@api.post("/analyse", response_model=AnalysisResponse)
async def analyse_company(request: AnalyseRequest):
    """Run the full competitive intelligence pipeline for a company."""
    start = time.time()
    company = request.company.strip()

    if not company:
        raise HTTPException(status_code=400, detail="Company name is required")

    try:
        # Import here to avoid circular import issues
        from projects.competitive_intel.agents import (
            OrchestratorAgent, MarketAgent, ProductAgent, TechStackAgent,
            NewsAgent, SentimentAgent, ConflictResolverAgent, SynthesisAgent,
        )
        from projects.competitive_intel.models import AgentReport

        orchestrator = OrchestratorAgent()
        specialists = {
            "market-agent": MarketAgent(),
            "product-agent": ProductAgent(),
            "tech-agent": TechStackAgent(),
            "news-agent": NewsAgent(),
            "sentiment-agent": SentimentAgent(),
        }
        conflict_resolver = ConflictResolverAgent()
        synthesis_agent = SynthesisAgent()

        # Plan
        requests = orchestrator.plan_research(company)

        # Research
        reports: list[AgentReport] = []
        for req in requests:
            target = req.context.get("target_agent", "")
            if target in specialists:
                if target == "market-agent":
                    reports.append(specialists[target].research(req))
                elif target == "product-agent":
                    reports.append(specialists[target].research(req))
                elif target == "tech-agent":
                    reports.append(specialists[target].research(req))
                elif target == "news-agent":
                    reports.append(specialists[target].research(req))
                elif target == "sentiment-agent":
                    reports.append(specialists[target].research(req))

        # Conflicts
        conflicts = conflict_resolver.detect_conflicts(reports)
        for c in conflicts:
            conflict_resolver.resolve_conflict(c, reports)

        # Synthesis
        synthesis = synthesis_agent.synthesise(company, reports, conflicts)

        elapsed = (time.time() - start) * 1000
        synthesis.total_processing_time_ms = round(elapsed, 1)

        # Save
        SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
        safe_name = company.lower().replace(" ", "_").replace(".", "")
        out_path = SAMPLE_DIR / f"report_{safe_name}.json"
        out_path.write_text(
            json.dumps(synthesis.model_dump(), indent=2, default=str),
            encoding="utf-8",
        )

        return AnalysisResponse(
            status="success",
            company=company,
            report=synthesis.model_dump(),
            processing_time_ms=round(elapsed, 1),
        )

    except Exception as e:
        elapsed = (time.time() - start) * 1000
        return AnalysisResponse(
            status="error",
            company=company,
            error=str(e)[:500],
            processing_time_ms=round(elapsed, 1),
        )


@api.get("/reports")
async def list_reports():
    """List all saved intelligence reports."""
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    reports = []
    for f in SAMPLE_DIR.glob("*.json"):
        reports.append({
            "filename": f.name,
            "size_bytes": f.stat().st_size,
        })
    return {"reports": reports, "count": len(reports)}


@api.get("/reports/{name}")
async def get_report(name: str):
    """Get a specific saved report by filename."""
    report_path = SAMPLE_DIR / name
    if not report_path.exists():
        raise HTTPException(status_code=404, detail=f"Report '{name}' not found")
    return json.loads(report_path.read_text(encoding="utf-8"))
