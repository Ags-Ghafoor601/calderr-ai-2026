# 🔬 Autonomous AI Research Lab

> **CalderR Internship — Week 5, Project 5-P-A (Production)**
>
> A multi-agent research system that dynamically assembles specialist agent teams
> based on domain classification, then executes a 5-phase research pipeline to
> produce peer-reviewed research papers.

## Overview

This project demonstrates **dynamic multi-agent orchestration** at production
scale. Unlike the Intermediate Project (fixed agent team), this system:

- **Classifies the research domain** (LLM + keyword fallback)
- **Dynamically assembles 3–8 agents** optimised for that domain
- **Executes 5 sequential phases** with typed handoffs between phases
- **Includes an adversarial Critic agent** that challenges every finding
- **Simulates academic peer review** with verdicts and quality scores

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  RESEARCH ORCHESTRATOR                      │
│  ┌───────────────────────────────────────────────────────┐  │
│  │ Domain Classifier → assembles 3–8 agents per domain  │  │
│  └───────────────────┬───────────────────────────────────┘  │
│                      │                                      │
│  ┌───────────────────▼───────────────────────────────────┐  │
│  │ Phase 1: HYPOTHESIS GENERATION                        │  │
│  │  • HypothesisGenerator (domain-specific prompts)      │  │
│  └───────────────────┬───────────────────────────────────┘  │
│  ┌───────────────────▼───────────────────────────────────┐  │
│  │ Phase 2: EVIDENCE GATHERING (fan-out)                 │  │
│  │  • LiteratureReviewer + DataAnalyst                   │  │
│  │  • MethodologyExpert + DomainSpecialist (if avail.)   │  │
│  └───────────────────┬───────────────────────────────────┘  │
│  ┌───────────────────▼───────────────────────────────────┐  │
│  │ Phase 3: CRITICAL ANALYSIS (Adversarial)              │  │
│  │  • CriticAgent — challenges hypotheses, evidence,     │  │
│  │    methodology, and biases                            │  │
│  └───────────────────┬───────────────────────────────────┘  │
│  ┌───────────────────▼───────────────────────────────────┐  │
│  │ Phase 4: SYNTHESIS                                    │  │
│  │  • SynthesisAgent → Abstract, Intro, Methods,         │  │
│  │    Findings, Discussion, Conclusion, Limitations      │  │
│  └───────────────────┬───────────────────────────────────┘  │
│  ┌───────────────────▼───────────────────────────────────┐  │
│  │ Phase 5: PEER REVIEW                                  │  │
│  │  • PeerReviewAgent → Accept / Minor / Major / Reject  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Supported Domains

| Domain | Specialist Agents | Total Team Size |
|--------|------------------|-----------------|
| **Technology** | Hypothesis, Literature, Data, Domain Specialist | 7 |
| **Medicine** | Hypothesis, Literature, Methodology, Domain Specialist, Data | 8 |
| **Economics** | Hypothesis, Literature, Data, Domain Specialist | 7 |
| **Environment** | Hypothesis, Literature, Data, Methodology | 7 |
| **Social Science** | Hypothesis, Literature, Methodology, Data | 7 |
| **General** | Hypothesis, Literature, Data | 6 |

All domains also include the 3 universal agents: **Critic**, **Synthesiser**, **Peer Reviewer**.

## Agent Roles

| Agent | Phase | Responsibility |
|-------|-------|----------------|
| **DomainClassifier** | 0 | Classifies topic → selects domain-specific team |
| **HypothesisGenerator** | 1 | Generates 3 testable hypotheses with novelty/relevance scores |
| **LiteratureReviewer** | 2 | Reviews papers, trials, industry reports |
| **DataAnalyst** | 2 | Quantitative evidence, statistics, benchmarks |
| **MethodologyExpert** | 2 | Evaluates study design, sampling, ethics |
| **DomainSpecialist** | 2 | Deep domain-specific expertise |
| **CriticAgent** | 3 | Adversarial — challenges all findings, detects bias |
| **SynthesisAgent** | 4 | Merges all into coherent research paper |
| **PeerReviewAgent** | 5 | Simulated academic review with verdict + score |

## Key Features

- **Dynamic agent assembly**: 3–8 agents selected per research domain
- **Domain classifier**: LLM-first classification with keyword fallback
- **5-phase pipeline**: Hypothesis → Evidence → Critique → Synthesis → Peer Review
- **Adversarial critique**: Dedicated critic agent challenges all findings
- **Peer review simulation**: Accept / Minor Revisions / Major Revisions / Reject
- **Quality scoring**: Weighted composite of rigor + confidence + review score
- **Typed Pydantic schemas**: All inter-phase handoffs use validated models
- **FastAPI REST API**: 5 endpoints with OpenAPI docs
- **Streamlit dashboard**: Real-time phase progress + report viewer
- **Docker Compose**: Multi-service deployment (API + Dashboard)
- **Batch mode**: Generate 5 reports across all domains

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
# Run demo with sample topic
python projects/research_lab/main.py demo

# Research a custom topic
python projects/research_lab/main.py research "Impact of AI on healthcare"

# Research with domain override
python projects/research_lab/main.py research "CRISPR gene therapy" --domain medicine

# Generate 5 reports across all domains
python projects/research_lab/main.py batch

# Show architecture
python projects/research_lab/main.py graph
```

### API Server

```bash
uvicorn projects.research_lab.api:api --reload --port 8001
```

Endpoints:
- `GET /health` — Health check
- `POST /research` — Run 5-phase pipeline (`{"topic": "...", "domain": "technology"}`)
- `GET /domains` — List all domains + their agent teams
- `GET /reports` — List saved reports
- `GET /reports/{filename}` — Get specific report

API docs at: `http://localhost:8001/docs`

### Streamlit Dashboard

```bash
streamlit run projects/research_lab/dashboard.py
```

### Docker Compose

```bash
docker-compose up --build
```

## Output Structure

Each research report includes:

```json
{
  "topic": "...",
  "domain": "technology",
  "hypothesis_report": { "hypotheses": [...], "methodology_suggestion": "..." },
  "evidence_report": { "evidence_items": [...], "literature_summary": "..." },
  "critique_report": { "critiques": [...], "overall_rigor_score": 0.75 },
  "synthesis_report": {
    "abstract": "...", "introduction": "...", "methodology": "...",
    "findings": "...", "discussion": "...", "conclusion": "...",
    "limitations": "...", "future_work": "...",
    "key_contributions": [...], "overall_confidence": 0.82
  },
  "peer_review_report": {
    "verdict": "minor_revisions", "overall_score": 0.78,
    "comments": [...], "strengths": [...], "weaknesses": [...],
    "recommendation": "..."
  },
  "agents_assembled": ["tech-hypothesis-agent", "tech-literature-agent", ...],
  "total_agents_used": 7,
  "phases_completed": 5,
  "overall_quality_score": 0.79
}
```

## Tech Stack

- **Python 3.12+**
- **Groq API** (LLaMA 3.1 8B) — LLM inference
- **Pydantic v2** — Typed schemas for all phases
- **FastAPI** — REST API with OpenAPI docs
- **Streamlit** — Interactive research dashboard
- **Rich** — CLI formatting with phase tracking
- **Typer** — CLI framework
- **Docker Compose** — Multi-service deployment

## Evaluation Criteria Met

- ✅ Dynamic agent assembly (3–8 agents per domain)
- ✅ 5-phase research pipeline with typed handoffs
- ✅ Adversarial critic agent challenges all findings
- ✅ Peer review with actionable verdict + quality score
- ✅ FastAPI returns structured JSON research reports
- ✅ Streamlit shows phase-by-phase progress
- ✅ Domain classifier with LLM + keyword fallback
- ✅ Docker Compose for multi-service deployment
- ✅ README includes full system design
- ✅ 5 research reports across different domains
- ✅ Architecture diagram included

---

*Built as part of CalderR Internship Week 5 — Multi-Agent Systems (Production)*
