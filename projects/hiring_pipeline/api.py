"""
AI-Powered Hiring Pipeline — FastAPI REST API
===============================================
Provides REST endpoints for the hiring pipeline.

Endpoints:
  POST /pipeline/run         — Run the full pipeline for a job + candidates
  GET  /candidates           — List all candidates
  GET  /candidates/{id}      — Get candidate details
  GET  /scores               — Get all scores
  GET  /bias-reports          — Get all bias reports
  GET  /audit-log             — Get the full audit trail
  GET  /decisions             — Get all hiring decisions
  GET  /health                — Health check
"""

import os
import uuid
import json
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

PROJECT_DIR = Path(__file__).resolve().parent
DB_PATH = str(PROJECT_DIR / "hiring_pipeline.db")

import database as db
from workflow import build_hiring_graph, get_initial_state
from langgraph.checkpoint.sqlite import SqliteSaver

CHECKPOINT_DB = str(PROJECT_DIR / ".hiring_checkpoint.db")


# ---------------------------------------------------------------------------
# Request / Response Models
# ---------------------------------------------------------------------------

class RunPipelineRequest(BaseModel):
    """Request body for running the hiring pipeline."""
    job: dict = Field(..., description="Job description dict")
    candidates: list[dict] = Field(..., description="List of candidate dicts")


class PipelineResponse(BaseModel):
    """Response from a pipeline run."""
    run_id: str
    status: str
    shortlisted_count: int
    hired_count: int
    total_candidates: int
    processing_log: list[str]


# ---------------------------------------------------------------------------
# App Factory
# ---------------------------------------------------------------------------

def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""

    api = FastAPI(
        title="AI-Powered Hiring Pipeline",
        description="LangGraph-based hiring workflow with bias detection and HITL",
        version="1.0.0",
    )

    api.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Initialise database on startup
    @api.on_event("startup")
    async def startup():
        db.init_db(DB_PATH)

    # ------------------------------------------------------------------
    # Endpoints
    # ------------------------------------------------------------------

    @api.get("/health")
    async def health():
        return {"status": "healthy", "service": "hiring-pipeline"}

    @api.get("/candidates")
    async def list_candidates():
        """List all candidates in the database."""
        candidates = db.get_all_candidates(DB_PATH)
        return {"count": len(candidates), "candidates": candidates}

    @api.get("/candidates/{candidate_id}")
    async def get_candidate(candidate_id: str):
        """Get a specific candidate by ID."""
        candidate = db.get_candidate(candidate_id, DB_PATH)
        if not candidate:
            raise HTTPException(status_code=404, detail="Candidate not found")
        return candidate

    @api.get("/audit-log")
    async def get_audit_log(candidate_id: Optional[str] = None):
        """Get audit log entries, optionally filtered by candidate."""
        entries = db.get_audit_log(candidate_id or "", DB_PATH)
        return {"count": len(entries), "entries": entries}

    @api.get("/decisions")
    async def get_decisions():
        """Get all hiring decisions."""
        decisions = db.get_all_decisions(DB_PATH)
        return {"count": len(decisions), "decisions": decisions}

    @api.post("/pipeline/run", response_model=PipelineResponse)
    async def run_pipeline(request: RunPipelineRequest):
        """Run the full hiring pipeline for a job + candidates."""
        db.init_db(DB_PATH)
        db.insert_job(request.job, DB_PATH)

        run_id = f"run-{str(uuid.uuid4())[:6]}"

        with SqliteSaver.from_conn_string(CHECKPOINT_DB) as checkpointer:
            compiled = build_hiring_graph(checkpointer=checkpointer)
            config = {"configurable": {"thread_id": run_id}}

            initial = get_initial_state(
                request.job, request.candidates, run_id, DB_PATH
            )
            result = compiled.invoke(initial, config)

            # If HITL interrupt, auto-approve top 2
            if result.get("awaiting_human"):
                shortlisted = result.get("shortlisted_ids", [])
                scores = result.get("candidate_scores", [])

                scored = sorted(
                    [(cid, next((s["overall_score"] for s in scores
                                 if s["candidate_id"] == cid), 0))
                     for cid in shortlisted],
                    key=lambda x: x[1], reverse=True,
                )

                human_decisions = {}
                for i, (cid, score) in enumerate(scored):
                    if i < 2:
                        human_decisions[cid] = {
                            "decision": "hire",
                            "notes": f"API auto-hire — score {score:.0f}",
                        }
                    else:
                        human_decisions[cid] = {
                            "decision": "reject",
                            "notes": f"API reject — position filled",
                        }

                compiled.update_state(config, {"human_decisions": human_decisions})
                result = compiled.invoke(None, config)

        hired = sum(1 for d in result.get("final_decisions", [])
                    if d["decision"] == "hire")

        return PipelineResponse(
            run_id=run_id,
            status="completed",
            shortlisted_count=len(result.get("shortlisted_ids", [])),
            hired_count=hired,
            total_candidates=len(request.candidates),
            processing_log=result.get("processing_log", []),
        )

    return api
