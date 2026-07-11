# API Aggregator Agent — Morning Briefing Generator

## 📋 Overview
An AI-powered agent that pulls data from **3 public APIs** in parallel and synthesizes a cohesive **morning briefing report** using Groq LLM. No external API keys required — all data sources are free and public.

## 🏗️ Architecture
```
Parallel Tool Calls (asyncio.gather)
  ├── 🌤️  Weather API   → wttr.in (no API key)
  ├── 📰  News API      → Hacker News (no API key)
  └── 💰  Finance API   → CoinGecko (no API key)
    ↓
Data Aggregator (merge all results)
    ↓
LLM Synthesizer (Groq + LangChain)
    ↓
Report Generator (HTML + Markdown)
```

## 🚀 Quick Start

### Prerequisites
- Python 3.10+
- Groq API key in `.env` file at project root

### Setup
```bash
# From the repo root
pip install -r requirements.txt

# Run the agent
python projects/api_aggregator/main.py

# Specify a different city
python projects/api_aggregator/main.py --city London

# Skip HTML report
python projects/api_aggregator/main.py --city Tokyo --no-html
```

### Output
The agent generates 3 files in `sample_reports/`:
- `briefing_YYYYMMDD_HHMMSS.md` — Markdown report
- `briefing_YYYYMMDD_HHMMSS.html` — Rich HTML report (open in browser)
- `briefing_YYYYMMDD_HHMMSS_raw.json` — Raw API data

## 🧰 Tools / Data Sources

| Tool | API | Key Required | Data |
|------|-----|-------------|------|
| `weather_tool` | wttr.in | ❌ No | Temperature, conditions, humidity, wind, UV |
| `news_tool` | Hacker News | ❌ No | Top stories, scores, comments, categories |
| `finance_tool` | CoinGecko | ❌ No | Crypto prices, 24h change, market cap |

## 📁 Project Structure
```
projects/api_aggregator/
├── main.py                # Entry point — CLI + pipeline orchestration
├── agent.py               # LangChain agent + parallel data fetching
├── report_generator.py    # HTML + Markdown report formatters
├── tools/
│   ├── __init__.py
│   ├── weather_tool.py    # wttr.in integration
│   ├── news_tool.py       # Hacker News API integration
│   └── finance_tool.py    # CoinGecko API integration
├── sample_reports/        # Generated reports saved here
├── requirements.txt       # Project-specific dependencies
└── README.md              # This file
```

## 🔑 Key Features
- **Parallel API calls** via `asyncio.gather()` — all 3 APIs fetched simultaneously
- **AI synthesis** via Groq LLM — raw data transformed into engaging narrative
- **Dual output formats** — HTML (beautiful dark theme) + Markdown (terminal/GitHub)
- **Robust error handling** — individual API failures don't crash the whole pipeline
- **Modular tools** — each API tool is independent and reusable
- **No API keys needed** — all data sources are free and public

## 📊 Skills Demonstrated
- Parallel tool calls with async Python
- API integration with httpx
- LangChain prompt engineering for data synthesis
- Report generation (HTML + Markdown)
- Error handling and graceful degradation
- CLI argument parsing
