"""
CalderR Internship – Week 5, Project 5-I-A
=============================================
Competitive Intelligence Agent — Streamlit Dashboard

Interactive UI showing:
  • Company input form
  • Real-time agent activity display
  • Intelligence report viewer
  • Agent architecture visualization

Run:
    streamlit run projects/competitive_intel/dashboard.py
"""

import os
import sys
import json
import time
from pathlib import Path

# Fix imports
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

import streamlit as st

PROJECT_DIR = Path(__file__).resolve().parent
SAMPLE_DIR = PROJECT_DIR / "sample_reports"

# Page config
st.set_page_config(
    page_title="Competitive Intelligence Agent",
    page_icon="🔍",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .agent-card {
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #333;
        margin: 0.5rem 0;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🔍 Competitive Intelligence Agent</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown("**Agent Architecture:**")
st.sidebar.markdown("""
```
Orchestrator
  ├── Market Agent
  ├── Product Agent
  ├── Tech Stack Agent
  ├── News Agent
  ├── Sentiment Agent
  ├── Conflict Resolver
  └── Synthesis Agent
```
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**Stack:** Python · Groq API · LangGraph · FastAPI")
st.sidebar.markdown("**Model:** LLaMA 3.1 8B")

# Main tabs
tab1, tab2, tab3 = st.tabs(["🔎 Analyse Company", "📊 Saved Reports", "🏗️ Architecture"])

with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        company_name = st.text_input(
            "Enter company name",
            placeholder="e.g., Tesla, OpenAI, Spotify...",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

    if run_button and company_name:
        st.markdown("---")

        # Agent status display
        st.subheader("🤖 Agent Activity")
        agent_cols = st.columns(5)
        agents_info = [
            ("📈 Market", "market-agent"),
            ("📦 Product", "product-agent"),
            ("💻 Tech Stack", "tech-agent"),
            ("📰 News", "news-agent"),
            ("💬 Sentiment", "sentiment-agent"),
        ]

        status_placeholders = {}
        for i, (label, key) in enumerate(agents_info):
            with agent_cols[i]:
                status_placeholders[key] = st.empty()
                status_placeholders[key].info(f"{label}\n⏳ Waiting...")

        # Progress bar
        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            from projects.competitive_intel.agents import (
                OrchestratorAgent, MarketAgent, ProductAgent, TechStackAgent,
                NewsAgent, SentimentAgent, ConflictResolverAgent, SynthesisAgent,
            )
            from projects.competitive_intel.models import AgentReport

            total_start = time.time()

            # Phase 1: Plan
            status_text.text("Phase 1: Planning research strategy...")
            orchestrator = OrchestratorAgent()
            requests = orchestrator.plan_research(company_name)
            progress_bar.progress(10)

            # Phase 2: Run specialists
            status_text.text("Phase 2: Running specialist agents...")
            specialists = {
                "market-agent": MarketAgent(),
                "product-agent": ProductAgent(),
                "tech-agent": TechStackAgent(),
                "news-agent": NewsAgent(),
                "sentiment-agent": SentimentAgent(),
            }

            reports: list[AgentReport] = []
            for idx, req in enumerate(requests):
                target = req.context.get("target_agent", "")
                label = agents_info[idx][0] if idx < len(agents_info) else target

                if target in status_placeholders:
                    status_placeholders[target].warning(f"{label}\n🔄 Running...")

                try:
                    if target == "market-agent":
                        report = specialists[target].research(req)
                    elif target == "product-agent":
                        report = specialists[target].research(req)
                    elif target == "tech-agent":
                        report = specialists[target].research(req)
                    elif target == "news-agent":
                        report = specialists[target].research(req)
                    elif target == "sentiment-agent":
                        report = specialists[target].research(req)
                    else:
                        continue

                    reports.append(report)
                    if target in status_placeholders:
                        status_placeholders[target].success(
                            f"{label}\n✅ Done ({report.confidence:.0%})"
                        )
                except Exception as e:
                    if target in status_placeholders:
                        status_placeholders[target].error(f"{label}\n❌ Failed")

                progress_bar.progress(10 + (idx + 1) * 15)

            # Phase 3: Conflict detection
            status_text.text("Phase 3: Detecting conflicts...")
            conflict_resolver = ConflictResolverAgent()
            conflicts = conflict_resolver.detect_conflicts(reports)
            for c in conflicts:
                conflict_resolver.resolve_conflict(c, reports)
            progress_bar.progress(90)

            # Phase 4: Synthesis
            status_text.text("Phase 4: Synthesising report...")
            synthesis_agent = SynthesisAgent()
            synthesis = synthesis_agent.synthesise(company_name, reports, conflicts)
            total_time = (time.time() - total_start) * 1000
            synthesis.total_processing_time_ms = round(total_time, 1)
            progress_bar.progress(100)
            status_text.text(f"✅ Complete! Total time: {total_time:.0f}ms")

            # Save report
            SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = company_name.lower().replace(" ", "_").replace(".", "")
            out_path = SAMPLE_DIR / f"report_{safe_name}.json"
            out_path.write_text(
                json.dumps(synthesis.model_dump(), indent=2, default=str),
                encoding="utf-8",
            )

            # Display report
            st.markdown("---")
            st.subheader(f"📋 Intelligence Report: {company_name}")

            # Executive Summary
            st.markdown("### Executive Summary")
            st.success(synthesis.executive_summary)

            # Metrics
            met_cols = st.columns(4)
            met_cols[0].metric("Confidence", f"{synthesis.overall_confidence:.0%}")
            met_cols[1].metric("Agents Used", synthesis.agents_used)
            met_cols[2].metric("Conflicts", len(synthesis.conflicts_detected))
            met_cols[3].metric("Time", f"{synthesis.total_processing_time_ms:.0f}ms")

            # Detailed sections
            st.markdown("### Detailed Analysis")
            detail_tabs = st.tabs(["📈 Market", "📦 Product", "💻 Technology", "📰 News", "💬 Sentiment"])

            with detail_tabs[0]:
                st.markdown(synthesis.market_analysis)
            with detail_tabs[1]:
                st.markdown(synthesis.product_analysis)
            with detail_tabs[2]:
                st.markdown(synthesis.technology_analysis)
            with detail_tabs[3]:
                st.markdown(synthesis.news_summary)
            with detail_tabs[4]:
                st.markdown(synthesis.sentiment_analysis)

            # Key Insights & Risks
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown("### 💡 Key Insights")
                for insight in synthesis.key_insights:
                    st.markdown(f"- {insight}")

            with col_b:
                st.markdown("### ⚠️ Risk Factors")
                for risk in synthesis.risk_factors:
                    st.markdown(f"- {risk}")

            # Recommendations
            if synthesis.recommendations:
                st.markdown("### 🎯 Recommendations")
                for i, rec in enumerate(synthesis.recommendations, 1):
                    st.markdown(f"{i}. {rec}")

            # Conflicts
            if synthesis.conflicts_detected:
                st.markdown("### ⚔️ Detected Conflicts")
                for c in synthesis.conflicts_detected:
                    with st.expander(f"Conflict: {c.get('topic', 'Unknown')[:50]}"):
                        st.write(f"**Agent A:** {c.get('agent_a', 'N/A')} — {c.get('claim_a', 'N/A')}")
                        st.write(f"**Agent B:** {c.get('agent_b', 'N/A')} — {c.get('claim_b', 'N/A')}")
                        st.write(f"**Resolution:** {c.get('resolution', 'Unresolved')}")

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")

    elif run_button and not company_name:
        st.warning("Please enter a company name.")

with tab2:
    st.subheader("📊 Saved Intelligence Reports")
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    reports_list = list(SAMPLE_DIR.glob("*.json"))

    if reports_list:
        for report_file in reports_list:
            with st.expander(f"📄 {report_file.stem}"):
                data = json.loads(report_file.read_text(encoding="utf-8"))
                st.json(data)
    else:
        st.info("No reports generated yet. Use the 'Analyse Company' tab to create one.")

with tab3:
    st.subheader("🏗️ System Architecture")
    st.markdown("""
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
               │     CONFLICT RESOLVER       │
               └──────────────┬──────────────┘
               ┌──────────────▼──────────────┐
               │      SYNTHESIS AGENT        │
               └──────────────┬──────────────┘
                        FINAL REPORT
    ```
    """)

    st.markdown("### Agent Roles")
    agent_data = {
        "Agent": ["Orchestrator", "Market Agent", "Product Agent", "Tech Stack Agent",
                   "News Agent", "Sentiment Agent", "Conflict Resolver", "Synthesis Agent"],
        "Role": [
            "Plans research strategy, creates sub-questions",
            "Market position, sizing, growth, competitors",
            "Core products, features, differentiators, weaknesses",
            "Technology choices, strengths, technical risks",
            "Recent developments, notable events",
            "Public/analyst sentiment, risk signals",
            "Detects and resolves contradictions between agents",
            "Merges all findings into executive briefing",
        ],
    }
    st.table(agent_data)
