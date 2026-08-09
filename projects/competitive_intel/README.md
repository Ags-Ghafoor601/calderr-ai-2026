# 🔍 Autonomous Competitive Intelligence Agent

> **CalderR Internship — Week 5, Project 5-I-A (Intermediate)**
>
> A multi-agent system that takes a company name and autonomously researches it
> from multiple angles, synthesising findings into a structured intelligence briefing.

## Overview

This project demonstrates **LangGraph fan-out/fan-in** architecture for parallel
multi-agent execution. Given a company name, the system orchestrates 5 specialist
agents that research different dimensions simultaneously, detects contradictions
between their findings, and produces a professional-grade intelligence report.

## System Architecture

```
┌──────────────────────┐
│  ORCHESTRATOR AGENT  │ ← Plans research, assigns sub-questions
└──────────┬───────────┘
           │ fan-out (parallel)
┌──────────▼──────────────────────────────────────┐
│              SPECIALIST AGENTS                   │
│  ┌────────┐ ┌────────┐ ┌──────┐ ┌────┐ ┌─────┐ │
│  │Market  │ │Product │ │ Tech │ │News│ │Sent.│ │
│  │Agent   │ │Agent   │ │Agent │ │Agt │ │Agt  │ │
│  └────┬───┘ └───┬────┘ └──┬───┘ └─┬──┘ └──┬──┘ │
└───────┼─────────┼────────┼───────┼───────┼────┘
        └─────────┴────────┴───────┴───────┘
                          │
           ┌──────────────▼──────────────┐
           │     CONFLICT RESOLVER       │ ← Detects contradictions
           └──────────────┬──────────────┘
           ┌──────────────▼──────────────┐
           │      SYNTHESIS AGENT        │ ← Merges into final briefing
           └──────────────┬──────────────┘
                    FINAL REPORT
```

## Agent Roles

| Agent | Responsibility |
|-------|----------------|
| **Orchestrator** | Plans research strategy, creates sub-questions for each specialist |
| **Market Agent** | Market position, sizing, growth trajectory, key competitors |
| **Product Agent** | Core products, features, differentiators, weaknesses |
| **Tech Stack Agent** | Inferred technologies, technical strengths, risks |
| **News Agent** | Recent developments, notable events, news sentiment |
| **Sentiment Agent** | Public/analyst sentiment, sentiment drivers, risk signals |
| **Conflict Resolver** | Detects contradictions between agents, arbitrates resolution |
| **Synthesis Agent** | Merges all findings into executive briefing with confidence scores |

## Key Features

- **Typed Pydantic communication**: All inter-agent messages use strict Pydantic schemas
- **Parallel agent execution**: 5 specialists run on the same task independently
- **Conflict detection & resolution**: LLM-powered contradiction detection with arbitration
- **Confidence scoring**: Every report includes a 0-1 confidence score
- **FastAPI REST API**: Programmatic access via HTTP endpoints
- **Streamlit dashboard**: Interactive UI with real-time agent activity display
- **3 sample reports**: Pre-generated intelligence reports on Tesla, OpenAI, and Spotify

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
```

### Environment

Create a `.env` file in the project root:

```
GROQ_API_KEY=your_groq_api_key
```

### CLI Usage

```bash
# Run demo with Tesla
python projects/competitive_intel/main.py demo

# Analyse a specific company
python projects/competitive_intel/main.py analyse "Microsoft"

# Generate and save a report
python projects/competitive_intel/main.py report "Spotify"

# Generate 3 sample reports
python projects/competitive_intel/main.py sample-reports

# Show architecture
python projects/competitive_intel/main.py graph
```

### API Server

```bash
uvicorn projects.competitive_intel.api:api --reload --port 8000
```

Endpoints:
- `GET /health` — Health check
- `POST /analyse` — Run intelligence pipeline (`{"company": "Tesla"}`)
- `GET /reports` — List saved reports
- `GET /reports/{name}` — Get specific report

API docs at: `http://localhost:8000/docs`

### Streamlit Dashboard

```bash
streamlit run projects/competitive_intel/dashboard.py
```

## Output Structure

Each intelligence report includes:

```json
{
  "company_name": "Tesla",
  "executive_summary": "...",
  "market_analysis": "...",
  "product_analysis": "...",
  "technology_analysis": "...",
  "news_summary": "...",
  "sentiment_analysis": "...",
  "key_insights": ["...", "..."],
  "risk_factors": ["...", "..."],
  "recommendations": ["...", "..."],
  "conflicts_detected": [],
  "overall_confidence": 0.85,
  "agents_used": 5,
  "total_processing_time_ms": 12500
}
```

## Tech Stack

- **Python 3.12+**
- **Groq API** (LLaMA 3.1 8B) — LLM inference
- **Pydantic v2** — Typed message schemas
- **FastAPI** — REST API
- **Streamlit** — Interactive dashboard
- **Rich** — CLI formatting
- **Typer** — CLI framework

## Evaluation Criteria Met

- ✅ All 5 research agents run and produce typed outputs
- ✅ Synthesis is coherent and non-redundant
- ✅ Conflicts detected and surfaced explicitly
- ✅ FastAPI returns valid JSON report
- ✅ Streamlit shows live agent status
- ✅ README includes full system design section
- ✅ 3 sample reports on real public companies
- ✅ Architecture diagram included

---

*Built as part of CalderR Internship Week 5 — Multi-Agent Systems*
