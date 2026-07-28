# 🏢 AI-Powered Hiring Pipeline — Project 4-P-A

A production-grade hiring workflow built with **LangGraph + FastAPI** that demonstrates complex graph orchestration, bias detection, human-in-the-loop approval, audit logging, and SQLite persistence.

## 🎯 Overview

This project implements a complete end-to-end hiring pipeline:
1. **Ingest** 10 sample resumes into a SQLite database
2. **Score** each candidate against the job description using an LLM
3. **Detect bias** in scoring (education prestige, age, name, consistency)
4. **Shortlist** top candidates above a configurable threshold
5. **Generate** tailored interview questions (technical, behavioral, situational)
6. **Human review** with interrupt/resume (HITL via SqliteSaver)
7. **Record decisions** and write a complete **audit trail**

## 🏗️ Architecture

```
    ┌─────────────────┐
    │  ingest_resumes  │  ← Register 10 candidates + DB insert
    └────────┬────────┘
    ┌────────▼────────┐
    │ score_candidates │  ← LLM scores each candidate (0-100)
    └────────┬────────┘
    ┌────────▼────────┐
    │   bias_check     │  ← Detect education/age/name bias
    └────────┬────────┘
    ┌────────▼────────┐
    │   shortlist      │  ← Top N above threshold (55/100)
    └────────┬────────┘
    ┌────────▼──────────────┐
    │ generate_questions     │  ← LLM: 3 questions per candidate
    └────────┬──────────────┘
    ┌────────▼────────┐
    │  human_review    │  ← HITL — interrupt for manager
    └────────┬────────┘
        ⏸️ INTERRUPT (SqliteSaver persistence)
    ┌────────▼────────┐
    │ apply_decisions  │  ← Apply hire / reject decisions
    └────────┬────────┘
    ┌────────▼────────┐
    │  final_audit     │  ← Summary audit log to SQLite
    └────────┬────────┘
             │
            END
```

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| **LangGraph** | 8-node graph workflow with HITL interrupt |
| **FastAPI** | REST API for programmatic access |
| **ChatGroq** (Llama 3.1) | Candidate scoring + question generation |
| **SQLite** | Candidates, scores, bias reports, audit log |
| **SqliteSaver** | Graph state persistence across HITL interrupts |
| **Pydantic** | Type-safe data models |
| **Rich + Typer** | Beautiful CLI with tables, trees, panels |

## 🚀 Quick Start

### Prerequisites
```bash
pip install -r requirements.txt
```

### Set API Key
Create a `.env` file in the project root (`Calder Internship/`):
```
GROQ_API_KEY=your_groq_api_key_here
```

### Run the Full Demo
```bash
python projects/hiring_pipeline/main.py demo
```

### View Audit Trail
```bash
python projects/hiring_pipeline/main.py audit
```

### View Bias Detection Report
```bash
python projects/hiring_pipeline/main.py bias-report
```

### View Graph Architecture
```bash
python projects/hiring_pipeline/main.py graph
```

### Start FastAPI Server
```bash
python projects/hiring_pipeline/main.py serve --port 8000
```

## 🌐 REST API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/health` | Health check |
| `POST` | `/pipeline/run` | Run full pipeline (job + candidates) |
| `GET` | `/candidates` | List all candidates |
| `GET` | `/candidates/{id}` | Get candidate details |
| `GET` | `/audit-log` | Full audit trail |
| `GET` | `/decisions` | All hiring decisions |

## 🔍 Bias Detection

The pipeline includes a dedicated bias detection node that checks for:

| Bias Category | What It Detects | Severity |
|--------------|-----------------|----------|
| **Education Prestige** | Inflated scores for elite universities | Medium |
| **Non-Traditional Ed** | Penalised bootcamp/certificate graduates | Medium |
| **Age (Junior)** | Low scores combined with "junior" indicators | Low |
| **Age (Senior)** | "Overqualified" bias for 12+ year candidates | Low |
| **Name Bias** | Visibility of names during scoring | Low |
| **Score Consistency** | Large gaps between component and overall scores | Low |

## 📊 Sample Candidates (10)

| # | Name | Experience | Education | Expected Outcome |
|---|------|-----------|-----------|-----------------|
| 1 | Amara Okafor | 6 yrs | BSc CompSci | Strong match — shortlisted |
| 2 | Liam Chen | 3 yrs | BSc SWE | Moderate — borderline |
| 3 | Priya Sharma | 5 yrs | MSc ML | Strong NLP — may mismatch |
| 4 | James O'Brien | 8 yrs | BSc IT | Java-heavy — moderate |
| 5 | Sofia Rodriguez | 1 yr | BSc CS | Junior — likely not shortlisted |
| 6 | Fatima Al-Hassan | 5 yrs | MSc CS | Strong match — shortlisted |
| 7 | Michael Thompson | 2 yrs | Bootcamp | Career changer — bias test |
| 8 | Yuki Tanaka | 4 yrs | PhD NLP | Strong NLP — different role fit |
| 9 | Oleksandr Kovalenko | 7 yrs | BSc CompEng | DevOps focus — partial match |
| 10 | Elizabeth Warren-Hughes | 15 yrs | MBA + MSc | Executive — overqualified test |

## 📁 Project Structure

```
hiring_pipeline/
├── main.py              # CLI entry point + demo
├── workflow.py          # LangGraph 8-node pipeline
├── models.py            # Pydantic data models
├── database.py          # SQLite CRUD layer
├── bias_detector.py     # Bias detection logic
├── sample_data.py       # 10 resumes + 2 job descriptions
├── api.py               # FastAPI REST endpoints
├── requirements.txt     # Dependencies
├── Dockerfile           # Docker deployment
└── README.md            # This file
```

## 📚 Skills Demonstrated

- Complex LangGraph workflows (8 nodes, conditional routing)
- Bias detection and fairness analysis
- Human-in-the-loop (HITL) with interrupt/resume
- Audit logging for compliance
- FastAPI integration with LangGraph
- SQLite persistence (database + checkpointer)
- LLM-powered scoring and question generation
- Production deployment with Docker
