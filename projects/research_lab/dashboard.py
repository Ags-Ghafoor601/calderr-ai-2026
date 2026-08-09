"""
CalderR Internship – Week 5, Project 5-P-A
=============================================
Autonomous AI Research Lab — Streamlit Dashboard

Interactive UI with:
  • Research topic input with domain auto-detection
  • Phase-by-phase progress tracking
  • Full report viewer with paper sections
  • Peer review verdict display
  • Agent team visualization
  • Saved reports browser

Run:
    streamlit run projects/research_lab/dashboard.py
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
REPORTS_DIR = PROJECT_DIR / "reports"

# Page config
st.set_page_config(
    page_title="AI Research Lab",
    page_icon="🔬",
    layout="wide",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 50%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .phase-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #00d2ff 0%, #3a7bd5 50%, #764ba2 100%);
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">🔬 Autonomous AI Research Lab</p>', unsafe_allow_html=True)
st.markdown("---")

# Sidebar
st.sidebar.header("⚙️ Research Configuration")
st.sidebar.markdown("**5-Phase Pipeline:**")
st.sidebar.markdown("""
1. 💡 Hypothesis Generation
2. 📚 Evidence Gathering
3. 🔍 Critical Analysis
4. 📝 Synthesis
5. 📋 Peer Review
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**Supported Domains:**")
st.sidebar.markdown("""
- 💻 Technology
- 🏥 Medicine
- 📈 Economics
- 🌍 Environment
- 👥 Social Science
- 📖 General
""")
st.sidebar.markdown("---")
st.sidebar.markdown("**Stack:** Python · Groq API · FastAPI · Pydantic v2")
st.sidebar.markdown("**Model:** LLaMA 3.1 8B Instant")

# Main tabs
tab1, tab2, tab3 = st.tabs(["🔬 Run Research", "📊 Saved Reports", "🏗️ Architecture"])

with tab1:
    col1, col2 = st.columns([3, 1])
    with col1:
        research_topic = st.text_input(
            "Enter research topic",
            placeholder="e.g., Impact of AI on scientific methodology...",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_button = st.button("🚀 Start Research", type="primary", use_container_width=True)

    if run_button and research_topic:
        st.markdown("---")

        # Phase progress display
        st.subheader("📡 Pipeline Progress")
        phase_cols = st.columns(5)
        phases_info = [
            ("💡 Hypothesis", "hypothesis"),
            ("📚 Evidence", "evidence"),
            ("🔍 Critique", "critique"),
            ("📝 Synthesis", "synthesis"),
            ("📋 Peer Review", "peer_review"),
        ]

        status_placeholders = {}
        for i, (label, key) in enumerate(phases_info):
            with phase_cols[i]:
                status_placeholders[key] = st.empty()
                status_placeholders[key].info(f"{label}\n⏳ Waiting...")

        progress_bar = st.progress(0)
        status_text = st.empty()

        try:
            from projects.research_lab.models import (
                ResearchDomain, FullResearchReport,
            )
            from projects.research_lab.domain_classifier import DomainClassifier
            from projects.research_lab.agents import (
                HypothesisGenerator, LiteratureReviewer, DataAnalyst,
                MethodologyExpert, DomainSpecialist, CriticAgent,
                SynthesisAgent as SynthesisAgentImpl, PeerReviewAgent,
            )
            from projects.research_lab.models import (
                AgentRole, EvidenceReport, EvidenceItem,
            )

            total_start = time.time()

            # Phase 0: Domain classification
            status_text.text("Classifying research domain...")
            classifier = DomainClassifier()
            domain = classifier.classify(research_topic)
            team = classifier.assemble_team(domain)
            agent_prompts = {a["role"]: a["system_prompt"] for a in team}
            progress_bar.progress(5)

            st.info(f"🏷️ Detected domain: **{domain.value}** | Assembled **{len(team)} agents**")

            # Phase 1: Hypothesis
            status_text.text("Phase 1: Generating hypotheses...")
            status_placeholders["hypothesis"].warning("💡 Hypothesis\n🔄 Running...")
            hyp_agent = HypothesisGenerator()
            hyp_prompt = agent_prompts.get(AgentRole.HYPOTHESIS_GENERATOR, team[0]["system_prompt"])
            hypothesis_report = hyp_agent.generate(research_topic, domain, hyp_prompt)
            status_placeholders["hypothesis"].success(
                f"💡 Hypothesis\n✅ {len(hypothesis_report.hypotheses)} hypotheses"
            )
            progress_bar.progress(20)

            # Phase 2: Evidence
            status_text.text("Phase 2: Gathering evidence...")
            status_placeholders["evidence"].warning("📚 Evidence\n🔄 Running...")
            all_evidence: list[EvidenceItem] = []
            lit_summary = ""
            data_summary = ""
            method_notes = ""
            agents_used = []

            lit_agents = [a for a in team if a["role"] == AgentRole.LITERATURE_REVIEWER]
            if lit_agents:
                lit_result = LiteratureReviewer().review(
                    research_topic, hypothesis_report.hypotheses, lit_agents[0]["system_prompt"]
                )
                all_evidence.extend(lit_result.get("evidence_items", []))
                lit_summary = lit_result.get("summary", "")
                agents_used.append(lit_agents[0]["name"])
            progress_bar.progress(35)

            data_agents = [a for a in team if a["role"] == AgentRole.DATA_ANALYST]
            if data_agents:
                data_result = DataAnalyst().analyse(
                    research_topic, hypothesis_report.hypotheses, data_agents[0]["system_prompt"]
                )
                all_evidence.extend(data_result.get("evidence_items", []))
                data_summary = data_result.get("analysis_summary", "")
                agents_used.append(data_agents[0]["name"])
            progress_bar.progress(50)

            method_agents = [a for a in team if a["role"] == AgentRole.METHODOLOGY_EXPERT]
            if method_agents:
                method_result = MethodologyExpert().evaluate(
                    research_topic, hypothesis_report.hypotheses, method_agents[0]["system_prompt"]
                )
                method_notes = method_result.get("methodology_review", "")
                agents_used.append(method_agents[0]["name"])

            spec_agents = [a for a in team if a["role"] == AgentRole.DOMAIN_SPECIALIST]
            if spec_agents:
                spec_result = DomainSpecialist().analyse(
                    research_topic, hypothesis_report.hypotheses, spec_agents[0]["system_prompt"]
                )
                all_evidence.extend(spec_result.get("evidence_items", []))
                agents_used.append(spec_agents[0]["name"])

            evidence_report = EvidenceReport(
                topic=research_topic,
                evidence_items=all_evidence,
                literature_summary=lit_summary,
                data_analysis_summary=data_summary,
                methodology_notes=method_notes,
                agents_used=agents_used,
            )
            status_placeholders["evidence"].success(
                f"📚 Evidence\n✅ {len(all_evidence)} items"
            )
            progress_bar.progress(60)

            # Phase 3: Critique
            status_text.text("Phase 3: Critical analysis...")
            status_placeholders["critique"].warning("🔍 Critique\n🔄 Running...")
            critic_agents = [a for a in team if a["role"] == AgentRole.CRITIC]
            critic_prompt = critic_agents[0]["system_prompt"] if critic_agents else "You are a research critic."
            critique_report = CriticAgent().critique(
                research_topic,
                hypothesis_report.model_dump(),
                evidence_report.model_dump(),
                critic_prompt,
            )
            status_placeholders["critique"].success(
                f"🔍 Critique\n✅ Rigor: {critique_report.overall_rigor_score:.0%}"
            )
            progress_bar.progress(75)

            # Phase 4: Synthesis
            status_text.text("Phase 4: Synthesising paper...")
            status_placeholders["synthesis"].warning("📝 Synthesis\n🔄 Running...")
            synth_agents = [a for a in team if a["role"] == AgentRole.SYNTHESISER]
            synth_prompt = synth_agents[0]["system_prompt"] if synth_agents else "Synthesise research."
            synthesis_report = SynthesisAgentImpl().synthesise(
                research_topic,
                hypothesis_report.model_dump(),
                evidence_report.model_dump(),
                critique_report.model_dump(),
                synth_prompt,
            )
            status_placeholders["synthesis"].success(
                f"📝 Synthesis\n✅ Confidence: {synthesis_report.overall_confidence:.0%}"
            )
            progress_bar.progress(90)

            # Phase 5: Peer Review
            status_text.text("Phase 5: Peer review...")
            status_placeholders["peer_review"].warning("📋 Peer Review\n🔄 Running...")
            review_agents = [a for a in team if a["role"] == AgentRole.PEER_REVIEWER]
            review_prompt = review_agents[0]["system_prompt"] if review_agents else "Peer review."
            peer_review = PeerReviewAgent().review(
                research_topic,
                synthesis_report.model_dump(),
                critique_report.model_dump(),
                review_prompt,
            )
            verdict_emoji = {
                "accept": "✅", "minor_revisions": "🟡",
                "major_revisions": "🟠", "reject": "🔴",
            }
            status_placeholders["peer_review"].success(
                f"📋 Peer Review\n{verdict_emoji.get(peer_review.verdict.value, '📋')} "
                f"{peer_review.verdict.value}"
            )
            progress_bar.progress(100)

            total_time = (time.time() - total_start) * 1000
            status_text.text(f"✅ Research complete! Total time: {total_time:.0f}ms")

            # Save
            REPORTS_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = research_topic[:30].lower().replace(" ", "_").replace("/", "_")
            out_path = REPORTS_DIR / f"report_{safe_name}.json"

            quality_score = (
                critique_report.overall_rigor_score * 0.3
                + synthesis_report.overall_confidence * 0.3
                + peer_review.overall_score * 0.4
            )

            report_data = {
                "topic": research_topic,
                "domain": domain.value,
                "hypothesis_report": hypothesis_report.model_dump(),
                "evidence_report": evidence_report.model_dump(),
                "critique_report": critique_report.model_dump(),
                "synthesis_report": synthesis_report.model_dump(),
                "peer_review_report": peer_review.model_dump(),
                "agents_assembled": [a["name"] for a in team],
                "total_agents_used": len(agents_used) + 3,
                "total_processing_time_ms": round(total_time, 1),
                "phases_completed": 5,
                "overall_quality_score": round(quality_score, 3),
                "status": "complete",
            }
            out_path.write_text(json.dumps(report_data, indent=2, default=str), encoding="utf-8")

            # Display report
            st.markdown("---")
            st.subheader(f"📋 Research Report: {research_topic}")

            # Metrics
            met_cols = st.columns(5)
            met_cols[0].metric("Domain", domain.value)
            met_cols[1].metric("Quality", f"{quality_score:.0%}")
            met_cols[2].metric("Agents", len(team))
            met_cols[3].metric("Verdict", peer_review.verdict.value)
            met_cols[4].metric("Time", f"{total_time:.0f}ms")

            # Paper sections
            st.markdown("### 📄 Research Paper")
            paper_tabs = st.tabs([
                "Abstract", "Introduction", "Methodology",
                "Findings", "Discussion", "Conclusion",
                "Limitations", "Future Work",
            ])

            synth_data = synthesis_report.model_dump()
            sections = [
                "abstract", "introduction", "methodology", "findings",
                "discussion", "conclusion", "limitations", "future_work",
            ]
            for i, section in enumerate(sections):
                with paper_tabs[i]:
                    st.markdown(synth_data.get(section, "N/A"))

            # Key contributions
            if synth_data.get("key_contributions"):
                st.markdown("### 💡 Key Contributions")
                for contrib in synth_data["key_contributions"]:
                    st.markdown(f"- {contrib}")

            # Peer Review
            st.markdown("### 📋 Peer Review")
            rev_col1, rev_col2 = st.columns(2)
            with rev_col1:
                st.markdown("**Strengths:**")
                for s in peer_review.strengths:
                    st.markdown(f"- ✅ {s}")
            with rev_col2:
                st.markdown("**Weaknesses:**")
                for w in peer_review.weaknesses:
                    st.markdown(f"- ⚠️ {w}")

            if peer_review.recommendation:
                st.info(f"**Recommendation:** {peer_review.recommendation}")

            # Hypotheses
            if hypothesis_report.hypotheses:
                st.markdown("### 💡 Generated Hypotheses")
                for i, h in enumerate(hypothesis_report.hypotheses, 1):
                    with st.expander(f"H{i}: {h.statement[:60]}..."):
                        st.write(f"**Statement:** {h.statement}")
                        st.write(f"**Rationale:** {h.rationale}")
                        st.write(f"**Testability:** {h.testability}")
                        st.write(f"**Novelty:** {h.novelty_score:.2f} | **Relevance:** {h.domain_relevance:.2f}")

            # Critiques
            if critique_report.critiques:
                st.markdown("### 🔍 Critical Issues Found")
                for c in critique_report.critiques:
                    severity_emoji = {"low": "🟢", "medium": "🟡", "high": "🟠", "critical": "🔴"}.get(c.severity.value, "⚪")
                    with st.expander(f"{severity_emoji} [{c.severity.value}] {c.issue[:50]}..."):
                        st.write(f"**Type:** {c.target_type}")
                        st.write(f"**Issue:** {c.issue}")
                        st.write(f"**Suggestion:** {c.suggestion}")

        except Exception as e:
            st.error(f"Research failed: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

    elif run_button and not research_topic:
        st.warning("Please enter a research topic.")

with tab2:
    st.subheader("📊 Saved Research Reports")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    reports_list = list(REPORTS_DIR.glob("*.json"))

    if reports_list:
        for report_file in reports_list:
            try:
                data = json.loads(report_file.read_text(encoding="utf-8"))
                topic = data.get("topic", report_file.stem)
                domain = data.get("domain", "unknown")
                quality = data.get("overall_quality_score", 0)
                with st.expander(f"📄 {topic} [{domain}] — Quality: {quality:.0%}"):
                    st.json(data)
            except Exception:
                with st.expander(f"📄 {report_file.stem}"):
                    st.json(json.loads(report_file.read_text(encoding="utf-8")))
    else:
        st.info("No reports generated yet. Use the 'Run Research' tab to create one.")

with tab3:
    st.subheader("🏗️ System Architecture")
    st.markdown("""
    ```
    ┌─────────────────────────────────────────────────────────────┐
    │                  RESEARCH ORCHESTRATOR                      │
    │  ┌───────────────────────────────────────────────────────┐  │
    │  │ Domain Classifier → assembles 3–8 agents per domain  │  │
    │  └───────────────────┬───────────────────────────────────┘  │
    │                      │                                      │
    │  ┌───────────────────▼───────────────────────────────────┐  │
    │  │ Phase 1: HYPOTHESIS GENERATION                        │  │
    │  │  • HypothesisGenerator (domain-specific)              │  │
    │  └───────────────────┬───────────────────────────────────┘  │
    │  ┌───────────────────▼───────────────────────────────────┐  │
    │  │ Phase 2: EVIDENCE GATHERING (fan-out)                 │  │
    │  │  • LiteratureReviewer + DataAnalyst                   │  │
    │  │  • MethodologyExpert + DomainSpecialist               │  │
    │  └───────────────────┬───────────────────────────────────┘  │
    │  ┌───────────────────▼───────────────────────────────────┐  │
    │  │ Phase 3: CRITICAL ANALYSIS (Adversarial)              │  │
    │  │  • CriticAgent — challenges everything                │  │
    │  └───────────────────┬───────────────────────────────────┘  │
    │  ┌───────────────────▼───────────────────────────────────┐  │
    │  │ Phase 4: SYNTHESIS                                    │  │
    │  │  • SynthesisAgent → full research paper               │  │
    │  └───────────────────┬───────────────────────────────────┘  │
    │  ┌───────────────────▼───────────────────────────────────┐  │
    │  │ Phase 5: PEER REVIEW                                  │  │
    │  │  • PeerReviewAgent → accept/minor/major/reject        │  │
    │  └───────────────────────────────────────────────────────┘  │
    └─────────────────────────────────────────────────────────────┘
    ```
    """)

    st.markdown("### Agent Roles by Domain")
    domain_data = {
        "Domain": ["Technology", "Medicine", "Economics", "Environment", "Social Science"],
        "Specialists": [
            "Hypothesis, Literature, Data, Specialist",
            "Hypothesis, Literature, Methodology, Specialist, Data",
            "Hypothesis, Literature, Data, Specialist",
            "Hypothesis, Literature, Data, Methodology",
            "Hypothesis, Literature, Methodology, Data",
        ],
        "Universal Agents": [
            "Critic, Synthesiser, Peer Reviewer",
            "Critic, Synthesiser, Peer Reviewer",
            "Critic, Synthesiser, Peer Reviewer",
            "Critic, Synthesiser, Peer Reviewer",
            "Critic, Synthesiser, Peer Reviewer",
        ],
        "Total Agents": ["7", "8", "7", "7", "7"],
    }
    st.table(domain_data)

    st.markdown("### 5-Phase Pipeline")
    pipeline_data = {
        "Phase": ["1. Hypothesis", "2. Evidence", "3. Critique", "4. Synthesis", "5. Peer Review"],
        "Purpose": [
            "Generate testable hypotheses",
            "Gather literature, data, expert opinions",
            "Adversarial review of all findings",
            "Merge into coherent research paper",
            "Simulated academic peer review",
        ],
        "Output": [
            "3 hypotheses with novelty/relevance scores",
            "Evidence items + summaries from 2-4 agents",
            "Critiques + bias warnings + rigor score",
            "Abstract, Intro, Methods, Findings, Discussion, Conclusion",
            "Verdict + score + strengths/weaknesses + recommendation",
        ],
    }
    st.table(pipeline_data)
