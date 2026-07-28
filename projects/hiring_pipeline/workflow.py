"""
AI-Powered Hiring Pipeline — LangGraph Workflow
=================================================
Complete hiring workflow: ingest → score → bias check → shortlist →
generate questions → human review (HITL) → decision → audit log.
"""

import os
import json
import hashlib
import time
from pathlib import Path
from typing import Annotated
from operator import add

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

from bias_detector import detect_bias
import database as db

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
SHORTLIST_THRESHOLD = 55.0     # Minimum score to be shortlisted
TOP_SHORTLIST = 5              # Max candidates on shortlist


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

def get_llm(temperature: float = 0.3):
    return ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=temperature,
        api_key=GROQ_API_KEY,
    )


# ---------------------------------------------------------------------------
# Pipeline State
# ---------------------------------------------------------------------------

class HiringState(TypedDict):
    """Graph state for the hiring pipeline.

    Holds ALL data flowing through the pipeline for a single hiring run
    (one job description + a batch of candidates).
    """
    # Job
    job_id: str
    job_title: str
    job_description: str
    required_skills: list[str]
    preferred_skills: list[str]
    min_experience: int
    education_requirement: str

    # Candidates (batch)
    candidates: list[dict]
    candidate_scores: list[dict]
    bias_reports: list[dict]
    shortlisted_ids: list[str]
    interview_questions: dict          # candidate_id → list of questions
    human_decisions: dict              # candidate_id → {decision, notes}
    final_decisions: list[dict]

    # HITL
    awaiting_human: bool
    pending_review_ids: list[str]

    # Meta
    processing_log: Annotated[list[str], add]
    run_id: str
    db_path: str


# ---------------------------------------------------------------------------
# Node: Ingest Resumes
# ---------------------------------------------------------------------------

def ingest_resumes(state: HiringState) -> dict:
    """Register candidates in the database."""
    candidates = state["candidates"]
    db_path = state["db_path"]

    for c in candidates:
        if not c.get("id"):
            c["id"] = "C-" + hashlib.sha256(
                f"{c['name']}-{c['email']}".encode()
            ).hexdigest()[:8].upper()
        c["status"] = "ingested"
        db.insert_candidate(c, db_path)
        db.log_audit(
            "INGEST", f"Candidate {c['name']} ingested",
            candidate_id=c["id"], job_id=state["job_id"],
            node_name="ingest_resumes", db_path=db_path,
        )

    return {
        "candidates": candidates,
        "processing_log": [
            f"[INGEST] 📥 {len(candidates)} candidates ingested into database"
        ],
    }


# ---------------------------------------------------------------------------
# Node: Score Candidates
# ---------------------------------------------------------------------------

def score_candidates(state: HiringState) -> dict:
    """Score each candidate against the job description using LLM."""
    llm = get_llm()
    candidates = state["candidates"]
    db_path = state["db_path"]
    scores = []

    required = ", ".join(state["required_skills"])
    preferred = ", ".join(state["preferred_skills"])

    for c in candidates:
        skills_str = ", ".join(c.get("skills", []))
        roles_str = "; ".join(c.get("previous_roles", []))

        prompt = f"""You are a hiring scoring system. Score this candidate against the job requirements.
Return ONLY valid JSON (no markdown, no code blocks).

JOB: {state['job_title']}
Required skills: {required}
Preferred skills: {preferred}
Min experience: {state['min_experience']} years
Education: {state['education_requirement']}

CANDIDATE:
Name: {c['name']}
Experience: {c['years_experience']} years
Education: {c['education']} from {c['university']}
Skills: {skills_str}
Roles: {roles_str}
Summary: {c['summary']}

Score each dimension 0-100. Be rigorous and fair.
Return JSON: {{"overall_score": N, "skills_match": N, "experience_match": N, "education_match": N, "summary_relevance": N, "scoring_rationale": "brief reason"}}"""

        try:
            result = llm.invoke([HumanMessage(content=prompt)])
            text = result.content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines).strip()
            parsed = json.loads(text)
            parsed["candidate_id"] = c["id"]
        except Exception as e:
            # Fallback: algorithmic scoring
            parsed = _algorithmic_score(c, state)

        # Clamp values
        for key in ["overall_score", "skills_match", "experience_match",
                     "education_match", "summary_relevance"]:
            parsed[key] = max(0.0, min(100.0, float(parsed.get(key, 0))))

        scores.append(parsed)
        db.insert_score(parsed, state["job_id"], db_path)
        db.update_candidate_status(c["id"], "scored", db_path)
        db.log_audit(
            "SCORE", f"Scored {c['name']}: {parsed['overall_score']:.0f}/100",
            candidate_id=c["id"], job_id=state["job_id"],
            node_name="score_candidates", db_path=db_path,
        )

    return {
        "candidate_scores": scores,
        "processing_log": [
            f"[SCORE] 📊 Scored {len(scores)} candidates — "
            f"Range: {min(s['overall_score'] for s in scores):.0f}"
            f"–{max(s['overall_score'] for s in scores):.0f}"
        ],
    }


def _algorithmic_score(candidate: dict, state: dict) -> dict:
    """Fallback algorithmic scoring when LLM fails."""
    cand_skills = {s.lower() for s in candidate.get("skills", [])}
    req_skills = {s.lower() for s in state.get("required_skills", [])}
    pref_skills = {s.lower() for s in state.get("preferred_skills", [])}

    req_match = len(cand_skills & req_skills) / max(len(req_skills), 1) * 100
    pref_match = len(cand_skills & pref_skills) / max(len(pref_skills), 1) * 100
    skills = req_match * 0.7 + pref_match * 0.3

    yrs = candidate.get("years_experience", 0)
    min_exp = state.get("min_experience", 0)
    if min_exp == 0:
        exp = 70.0
    elif yrs >= min_exp:
        exp = min(100, 70 + (yrs - min_exp) * 5)
    else:
        exp = max(0, 70 - (min_exp - yrs) * 15)

    edu = 50.0  # Neutral default
    education = (candidate.get("education") or "").lower()
    if "ph.d" in education or "phd" in education:
        edu = 90.0
    elif "m.sc" in education or "master" in education or "mba" in education:
        edu = 80.0
    elif "b.sc" in education or "bachelor" in education:
        edu = 65.0

    overall = skills * 0.45 + exp * 0.30 + edu * 0.15 + 50 * 0.10  # 10% baseline

    return {
        "candidate_id": candidate.get("id", ""),
        "overall_score": round(overall, 1),
        "skills_match": round(skills, 1),
        "experience_match": round(exp, 1),
        "education_match": round(edu, 1),
        "summary_relevance": 50.0,
        "scoring_rationale": "Algorithmic fallback scoring",
    }


# ---------------------------------------------------------------------------
# Node: Bias Check
# ---------------------------------------------------------------------------

def bias_check(state: HiringState) -> dict:
    """Run bias detection on all candidate scores."""
    candidates = state["candidates"]
    scores = state["candidate_scores"]
    db_path = state["db_path"]
    reports = []

    job_dict = {
        "required_skills": state["required_skills"],
        "preferred_skills": state["preferred_skills"],
        "min_experience": state["min_experience"],
        "education_requirement": state["education_requirement"],
    }

    for c in candidates:
        c_score = next((s for s in scores if s["candidate_id"] == c["id"]), {})
        report = detect_bias(c, c_score, job_dict)
        reports.append(report)
        db.insert_bias_report(report, state["job_id"], db_path)
        db.update_candidate_status(c["id"], "bias_checked", db_path)

        flag_count = len(report["flags"])
        risk = report["overall_risk"]
        db.log_audit(
            "BIAS_CHECK",
            f"Bias check for {c['name']}: {flag_count} flags, risk={risk}",
            candidate_id=c["id"], job_id=state["job_id"],
            node_name="bias_check", db_path=db_path,
        )

    high_risk = sum(1 for r in reports if r["overall_risk"] == "high")
    med_risk = sum(1 for r in reports if r["overall_risk"] == "medium")

    return {
        "bias_reports": reports,
        "processing_log": [
            f"[BIAS] 🔍 Checked {len(reports)} candidates — "
            f"High risk: {high_risk}, Medium: {med_risk}, "
            f"Low: {len(reports) - high_risk - med_risk}"
        ],
    }


# ---------------------------------------------------------------------------
# Node: Shortlist
# ---------------------------------------------------------------------------

def shortlist_candidates(state: HiringState) -> dict:
    """Select top candidates based on (bias-adjusted) scores."""
    scores = state["candidate_scores"]
    reports = state["bias_reports"]
    db_path = state["db_path"]

    # Use adjusted score from bias report where available
    adjusted = {}
    for r in reports:
        adjusted[r["candidate_id"]] = r.get("adjusted_score", 0)

    scored_list = []
    for s in scores:
        cid = s["candidate_id"]
        adj = adjusted.get(cid, s["overall_score"])
        scored_list.append((cid, adj, s["overall_score"]))

    # Sort by adjusted score descending
    scored_list.sort(key=lambda x: x[1], reverse=True)

    # Select above threshold, capped at TOP_SHORTLIST
    shortlisted = [
        cid for cid, adj, _ in scored_list
        if adj >= SHORTLIST_THRESHOLD
    ][:TOP_SHORTLIST]

    # Update statuses
    for c in state["candidates"]:
        if c["id"] in shortlisted:
            db.update_candidate_status(c["id"], "shortlisted", db_path)
            db.log_audit(
                "SHORTLIST", f"{c['name']} SHORTLISTED (score ≥ {SHORTLIST_THRESHOLD})",
                candidate_id=c["id"], job_id=state["job_id"],
                node_name="shortlist", db_path=db_path,
            )
        else:
            db.update_candidate_status(c["id"], "not_shortlisted", db_path)
            db.log_audit(
                "SHORTLIST", f"{c['name']} not shortlisted",
                candidate_id=c["id"], job_id=state["job_id"],
                node_name="shortlist", db_path=db_path,
            )

    return {
        "shortlisted_ids": shortlisted,
        "processing_log": [
            f"[SHORTLIST] ✂️ {len(shortlisted)}/{len(scores)} candidates "
            f"shortlisted (threshold: {SHORTLIST_THRESHOLD})"
        ],
    }


# ---------------------------------------------------------------------------
# Node: Generate Interview Questions
# ---------------------------------------------------------------------------

def generate_questions(state: HiringState) -> dict:
    """Generate tailored interview questions for shortlisted candidates."""
    llm = get_llm(temperature=0.5)
    shortlisted = state["shortlisted_ids"]
    candidates = state["candidates"]
    scores = state["candidate_scores"]
    db_path = state["db_path"]
    questions_map = {}

    for cid in shortlisted:
        c = next((x for x in candidates if x["id"] == cid), None)
        s = next((x for x in scores if x["candidate_id"] == cid), None)
        if not c:
            continue

        prompt = f"""Generate 3 interview questions for this candidate applying for {state['job_title']}.

Candidate: {c['name']}
Skills: {', '.join(c.get('skills', []))}
Experience: {c['years_experience']} years
Scoring rationale: {s.get('scoring_rationale', '') if s else ''}

Generate exactly 3 questions (1 technical, 1 behavioral, 1 situational).
Return ONLY valid JSON (no markdown, no code blocks):
[{{"category": "technical|behavioral|situational", "question": "...", "follow_up": "...", "evaluation_criteria": "..."}}]"""

        try:
            result = llm.invoke([HumanMessage(content=prompt)])
            text = result.content.strip()
            if text.startswith("```"):
                lines = text.split("\n")
                lines = [l for l in lines if not l.strip().startswith("```")]
                text = "\n".join(lines).strip()
            qs = json.loads(text)
            if not isinstance(qs, list):
                qs = [qs]
        except Exception:
            qs = [
                {"category": "technical", "question": f"Describe your experience with {state['required_skills'][0] if state['required_skills'] else 'Python'}.", "follow_up": "What challenges did you face?", "evaluation_criteria": "Depth of knowledge"},
                {"category": "behavioral", "question": "Tell me about a time you resolved a team conflict.", "follow_up": "What would you do differently?", "evaluation_criteria": "Communication and leadership"},
                {"category": "situational", "question": f"If tasked to build a {state['job_title'].lower()} system from scratch, how would you approach it?", "follow_up": "What trade-offs would you consider?", "evaluation_criteria": "Problem-solving and design thinking"},
            ]

        questions_map[cid] = qs
        db.update_candidate_status(cid, "questions_generated", db_path)
        db.log_audit(
            "QUESTIONS", f"Generated {len(qs)} interview questions for {c['name']}",
            candidate_id=cid, job_id=state["job_id"],
            node_name="generate_questions", db_path=db_path,
        )

    return {
        "interview_questions": questions_map,
        "processing_log": [
            f"[QUESTIONS] ❓ Generated interview questions for "
            f"{len(questions_map)} shortlisted candidates"
        ],
    }


# ---------------------------------------------------------------------------
# Node: Human Review (HITL interrupt point)
# ---------------------------------------------------------------------------

def human_review(state: HiringState) -> dict:
    """Flag shortlisted candidates for human review."""
    shortlisted = state["shortlisted_ids"]
    db_path = state["db_path"]

    for cid in shortlisted:
        db.update_candidate_status(cid, "pending_review", db_path)
        db.log_audit(
            "HUMAN_REVIEW", f"Candidate {cid} queued for human review",
            candidate_id=cid, job_id=state["job_id"],
            node_name="human_review", db_path=db_path,
        )

    return {
        "awaiting_human": True,
        "pending_review_ids": shortlisted,
        "processing_log": [
            f"[REVIEW] ⏸️ {len(shortlisted)} candidates queued for human review"
        ],
    }


# ---------------------------------------------------------------------------
# Node: Apply Human Decisions
# ---------------------------------------------------------------------------

def apply_decisions(state: HiringState) -> dict:
    """Apply human decisions (hire / reject / hold) to candidates."""
    decisions_input = state.get("human_decisions", {})
    candidates = state["candidates"]
    scores = state["candidate_scores"]
    bias_reports = state["bias_reports"]
    questions = state.get("interview_questions", {})
    db_path = state["db_path"]
    final = []

    for cid in state.get("shortlisted_ids", []):
        c = next((x for x in candidates if x["id"] == cid), None)
        s = next((x for x in scores if x["candidate_id"] == cid), None)
        br = next((x for x in bias_reports if x["candidate_id"] == cid), None)
        qs = questions.get(cid, [])
        human = decisions_input.get(cid, {})

        decision = human.get("decision", "reject")
        notes = human.get("notes", "")
        decided_by = "human" if human else "auto"

        status = "hired" if decision == "hire" else "rejected"
        db.update_candidate_status(cid, status, db_path)

        dec = {
            "candidate_id": cid,
            "decision": decision,
            "decided_by": decided_by,
            "rationale": notes or f"{'Hired' if decision == 'hire' else 'Rejected'} — score {s.get('overall_score', 0):.0f}",
            "questions": qs,
            "bias_report": br or {},
            "score_data": s or {},
        }
        final.append(dec)
        db.insert_decision(dec, state["job_id"], db_path)
        db.log_audit(
            "DECISION",
            f"{'✅ HIRE' if decision == 'hire' else '❌ REJECT'}: "
            f"{c['name'] if c else cid} — {notes}",
            candidate_id=cid, job_id=state["job_id"],
            decision_by=decided_by, node_name="apply_decisions",
            db_path=db_path,
        )

    return {
        "final_decisions": final,
        "awaiting_human": False,
        "processing_log": [
            f"[DECISION] 📋 Applied decisions for {len(final)} candidates — "
            f"Hired: {sum(1 for d in final if d['decision'] == 'hire')}, "
            f"Rejected: {sum(1 for d in final if d['decision'] != 'hire')}"
        ],
    }


# ---------------------------------------------------------------------------
# Node: Final Audit Log
# ---------------------------------------------------------------------------

def final_audit(state: HiringState) -> dict:
    """Write final summary audit entry."""
    db_path = state["db_path"]
    total = len(state["candidates"])
    shortlisted = len(state.get("shortlisted_ids", []))
    hired = sum(1 for d in state.get("final_decisions", []) if d["decision"] == "hire")

    db.log_audit(
        "PIPELINE_COMPLETE",
        f"Pipeline complete: {total} candidates → {shortlisted} shortlisted → {hired} hired",
        job_id=state["job_id"], node_name="final_audit", db_path=db_path,
    )

    return {
        "processing_log": [
            f"[AUDIT] 📝 Pipeline complete — "
            f"{total} ingested → {shortlisted} shortlisted → {hired} hired"
        ],
    }


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_hiring_graph(checkpointer=None):
    """Build the complete hiring pipeline graph."""
    graph = StateGraph(HiringState)

    graph.add_node("ingest_resumes", ingest_resumes)
    graph.add_node("score_candidates", score_candidates)
    graph.add_node("bias_check", bias_check)
    graph.add_node("shortlist", shortlist_candidates)
    graph.add_node("generate_questions", generate_questions)
    graph.add_node("human_review", human_review)
    graph.add_node("apply_decisions", apply_decisions)
    graph.add_node("final_audit", final_audit)

    graph.set_entry_point("ingest_resumes")

    graph.add_edge("ingest_resumes", "score_candidates")
    graph.add_edge("score_candidates", "bias_check")
    graph.add_edge("bias_check", "shortlist")
    graph.add_edge("shortlist", "generate_questions")
    graph.add_edge("generate_questions", "human_review")
    graph.add_edge("human_review", "apply_decisions")
    graph.add_edge("apply_decisions", "final_audit")
    graph.add_edge("final_audit", END)

    return graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["apply_decisions"],
    )


def get_initial_state(job: dict, candidates: list[dict],
                      run_id: str = "", db_path: str = "") -> dict:
    """Create the initial pipeline state."""
    return {
        "job_id": job.get("id", ""),
        "job_title": job.get("title", ""),
        "job_description": job.get("description", ""),
        "required_skills": job.get("required_skills", []),
        "preferred_skills": job.get("preferred_skills", []),
        "min_experience": job.get("min_experience", 0),
        "education_requirement": job.get("education_requirement", ""),
        "candidates": candidates,
        "candidate_scores": [],
        "bias_reports": [],
        "shortlisted_ids": [],
        "interview_questions": {},
        "human_decisions": {},
        "final_decisions": [],
        "awaiting_human": False,
        "pending_review_ids": [],
        "processing_log": [],
        "run_id": run_id,
        "db_path": db_path or str(db.DB_PATH),
    }
