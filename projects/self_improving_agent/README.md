# Procedural Memory & Self-Improving Agent

> **CalderR Internship — Week 6, Intermediate Project (6-I-C)**
> An AI agent that learns from its own mistakes through procedural memory.

## Overview

This project builds an agent that **learns from corrections**. When a user corrects an answer, the agent:
1. **Extracts** the correction as a generalised rule (procedural memory)
2. **Stores** it in a SQLite rule store with domain classification and confidence scoring
3. **Retrieves** relevant rules at generation time using LLM-based matching
4. **Applies** the rules to future responses via augmented prompting
5. **Tracks** its own performance and plots a learning curve

Over 20 interactions, the agent's error rate **measurably decreases**.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                  SELF-IMPROVING AGENT                                │
│                                                                     │
│  USER INPUT                                                         │
│      │                                                              │
│      ▼                                                              │
│  ┌──────────────────┐                                               │
│  │  RULE RETRIEVER  │◄─── Queries SQLite rule store                 │
│  │  (LLM-based      │     by semantic similarity                    │
│  │   matching)       │     to current query                         │
│  └────────┬─────────┘                                               │
│           │ relevant rules                                          │
│           ▼                                                         │
│  ┌──────────────────────┐                                           │
│  │  RULE-AUGMENTED      │                                           │
│  │  PROMPT BUILDER      │                                           │
│  │  ────────────────    │                                           │
│  │  Base prompt +       │                                           │
│  │  injected rules      │                                           │
│  └────────┬─────────────┘                                           │
│           │                                                         │
│           ▼                                                         │
│  ┌──────────────────┐                                               │
│  │  GROQ LLM        │───► RESPONSE                                 │
│  │  (rule-aware)     │                                               │
│  └──────────────────┘                                               │
│           │                                                         │
│           ▼                                                         │
│  ┌──────────────────┐      ┌─────────────────────┐                  │
│  │  USER FEEDBACK   │─────►│  CORRECTION         │                  │
│  │  HANDLER         │      │  EXTRACTOR AGENT    │                  │
│  │  ─────────────   │      │  ─────────────────  │                  │
│  │  If correction:  │      │  Extracts general   │                  │
│  │  extract rule    │      │  rule from specific  │                  │
│  │  If accepted:    │      │  correction          │                  │
│  │  log success     │      └──────────┬──────────┘                  │
│  └──────────────────┘                 │                             │
│                                       ▼                             │
│                           ┌───────────────────────┐                 │
│                           │  RULE STORE (SQLite)  │                 │
│                           │  ──────────────────   │                 │
│                           │  rule_text            │                 │
│                           │  domain               │                 │
│                           │  confidence           │                 │
│                           │  application_count    │                 │
│                           │  original_mistake     │                 │
│                           └───────────┬───────────┘                 │
│                                       │                             │
│                           ┌───────────▼───────────┐                 │
│                           │  RULE CONSOLIDATION   │                 │
│                           │  ──────────────────   │                 │
│                           │  3+ similar rules →   │                 │
│                           │  merge into one       │                 │
│                           │  high-confidence rule │                 │
│                           └───────────┬───────────┘                 │
│                                       │                             │
│                           ┌───────────▼───────────┐                 │
│                           │  PERFORMANCE TRACKER  │                 │
│                           │  ──────────────────   │                 │
│                           │  was_correct          │                 │
│                           │  quality_score        │                 │
│                           │  rules_applied_count  │                 │
│                           └───────────┬───────────┘                 │
│                                       │                             │
│                           ┌───────────▼───────────┐                 │
│                           │  LEARNING CURVE       │                 │
│                           │  VISUALISER           │                 │
│                           │  (matplotlib)         │                 │
│                           └───────────────────────┘                 │
└─────────────────────────────────────────────────────────────────────┘
```

## Memory Design

### Procedural Store (SQLite)

| Column | Type | Description |
|--------|------|-------------|
| `rule_id` | TEXT PK | Unique rule identifier |
| `original_mistake` | TEXT | What the agent got wrong |
| `correction` | TEXT | User's correction text |
| `rule_text` | TEXT | Generalised rule for future use |
| `domain` | TEXT | Category: factual, formatting, tone, reasoning, accuracy, completeness |
| `confidence` | REAL | 0.0–1.0, boosted by consolidation |
| `application_count` | INT | Times this rule was applied |
| `last_applied` | TEXT | ISO timestamp of last application |
| `is_active` | INT | 1=active, 0=merged/deactivated |

### Rule Consolidation
After 3 identical (case-insensitive) corrections in the same domain:
- The highest-confidence rule absorbs the others
- Its confidence increases by `+0.1 × (merge_count - 1)`
- Merged rules are marked `is_active = 0`

## Project Structure

```
self_improving_agent/
├── main.py           # Streamlit dashboard (entry point)
├── agent.py          # Core self-improving agent
├── memory.py         # SQLite procedural memory store
├── models.py         # Pydantic data schemas
├── evaluator.py      # Learning curve computation & evaluation
├── requirements.txt  # Python dependencies
└── README.md         # This file
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the Streamlit dashboard
streamlit run projects/self_improving_agent/main.py

# Or run the 20-interaction evaluation directly
python projects/self_improving_agent/evaluator.py
```

## Evaluation

The 20-interaction evaluation demonstrates measurable improvement:

- **Interactions 1–5**: Agent encounters common mistake patterns, user provides corrections → rules are extracted
- **Interactions 6–10**: Agent faces similar questions → learned rules are applied → fewer mistakes
- **Interactions 11–15**: New corrections for edge cases → rules expand
- **Interactions 16–20**: Agent applies full rule book → error rate is measurably lower

### Key Metrics
- **Early error rate (interactions 1-5)**: ~60-80%
- **Late error rate (interactions 16-20)**: ~0-20%
- **Improvement**: 40-80% error rate reduction
- **Rules learned**: 6-8 generalised correction rules

## Deliverables

- [x] GitHub repo with system design README
- [x] Streamlit dashboard with live rule book, correction interface, and learning curve chart
- [x] 20-interaction demonstration dataset showing error rate improvement
- [x] Rule extraction quality analysis
- [x] Before/after comparison of agent behaviour
- [x] Architecture diagram
- [x] Learning curve chart committed to repo
