# 🏢 Customer Onboarding Agent — Project 4-I-B

A multi-step customer onboarding workflow built with **LangGraph** that demonstrates conditional routing, human-in-the-loop approval, and state persistence.

## 🎯 Overview

This project implements a complete customer onboarding pipeline where:
- **Starter & Professional accounts** are auto-approved
- **Enterprise accounts** (>100 employees or >$1M revenue) require human approval
- State persists across interrupts using **SqliteSaver**
- LLM generates personalized welcome emails
- Follow-up meetings are scheduled based on tier

## 🏗️ Architecture

```
    ┌───────────────┐
    │  collect_info  │  ← Register customer data
    └───────┬───────┘
            │
    ┌───────▼───────┐
    │   validate     │  ← Check fields, email, business rules
    └───────┬───────┘
            │
     ┌──────▼──────┐
     │  determine   │  ← Starter / Professional / Enterprise
     │    tier      │
     └──────┬──────┘
            │
    ┌───────▼────────┐
    │  needs human   │
    │  approval?     │
    └───┬────────┬───┘
   yes  │        │ no
  ┌─────▼────┐ ┌─▼──────────┐
  │  human   │ │   auto     │
  │  review  │ │  approve   │
  └─────┬────┘ └─┬──────────┘
  ⏸️ INT │        │
  ┌─────▼────┐   │
  │  apply   │   │
  │ decision │   │
  └─────┬────┘   │
        └────┬───┘
    ┌────────▼────────┐
    │ create_account  │  ← Generate ID + API key
    └────────┬────────┘
    ┌────────▼────────┐
    │ send_welcome    │  ← LLM-generated welcome email
    └────────┬────────┘
    ┌────────▼──────────┐
    │schedule_followup  │  ← Based on tier (2/5/7 days)
    └────────┬──────────┘
             │
            END
```

## 🛠️ Tech Stack

| Technology | Purpose |
|-----------|---------|
| LangGraph | Graph-based workflow orchestration |
| SqliteSaver | State persistence across interrupts |
| ChatGroq (Llama 3.1) | Welcome email generation |
| Pydantic | Data validation models |
| Rich + Typer | Beautiful CLI interface |

## 🚀 Quick Start

### Prerequisites
```bash
pip install langgraph langgraph-checkpoint-sqlite langchain-groq rich typer pydantic python-dotenv
```

### Set API Key
Create a `.env` file in the project root:
```
GROQ_API_KEY=your_groq_api_key_here
```

### Run Demo
```bash
python projects/customer_onboarding/main.py demo
```

### Onboard a Single Customer
```bash
python projects/customer_onboarding/main.py onboard \
  --name "Acme Corp" \
  --contact "John Smith" \
  --email "john@acme.com" \
  --size 50 \
  --revenue 500000 \
  --industry "Technology" \
  --use-case "AI workflow automation"
```

### Approve/Reject Enterprise Accounts
```bash
# After an enterprise account triggers human review:
python projects/customer_onboarding/main.py approve <thread-id>
python projects/customer_onboarding/main.py reject <thread-id> --reason "Needs more info"
```

### View Graph Structure
```bash
python projects/customer_onboarding/main.py graph
```

## 📊 Account Tiers

| Tier | Criteria | Approval | Follow-up |
|------|----------|----------|-----------|
| 🌱 Starter | <10 employees, <$100K revenue | Auto | 7 days |
| 💼 Professional | 10-100 employees or $100K-$1M | Auto | 5 days |
| 🏢 Enterprise | >100 employees or >$1M revenue | **Human Required** | 2 days |

## 🔑 Key Features

- **Conditional Routing**: Three-way branching based on account tier
- **Human-in-the-Loop**: Enterprise accounts pause for manager approval
- **State Persistence**: SqliteSaver checkpoints survive process restarts
- **Data Validation**: Comprehensive field checks with error reporting
- **LLM Integration**: Personalized welcome emails via Groq
- **Audit Logging**: All decisions logged with timestamps

## 📁 Project Structure

```
customer_onboarding/
├── main.py              # LangGraph workflow + CLI
├── models.py            # Pydantic data models
├── README.md            # This file
├── onboarding_log.json  # Generated: audit log
└── .onboarding_checkpoint.db  # Generated: SQLite state
```

## 📚 Skills Learned

- Workflow design with LangGraph StateGraph
- Human-in-the-loop interrupt/resume patterns
- Conditional routing with multiple branch paths
- State persistence with SqliteSaver
- TypedDict state schemas with Annotated reducers
- Pydantic data validation integration
