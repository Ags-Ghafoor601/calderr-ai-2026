"""
Procedural Memory & Self-Improving Agent — Streamlit Dashboard
==============================================================
Interactive dashboard with:
  • Live chat with correction interface
  • Rule book viewer (all learned rules)
  • Learning curve chart
  • Performance metrics
  • Before/after comparison

Run:
    streamlit run projects/self_improving_agent/main.py
"""

import os
import sys
import json
import time
from pathlib import Path

# Add project root to path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

ROOT_DIR = PROJECT_DIR.parent.parent

import streamlit as st

from models import CorrectionRule, RuleDomain
from memory import ProceduralMemoryStore
from agent import SelfImprovingAgent
from evaluator import compute_learning_curve, generate_learning_curve_chart, run_evaluation

# ─── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Self-Improving Agent — Procedural Memory",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp {
        background-color: #0e1117;
    }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin-bottom: 10px;
    }
    .metric-value {
        font-size: 2.2rem;
        font-weight: bold;
        color: #58a6ff;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #8b949e;
        margin-top: 5px;
    }
    .rule-card {
        background: #161b22;
        border: 1px solid #30363d;
        border-radius: 10px;
        padding: 15px;
        margin-bottom: 10px;
    }
    .rule-domain {
        display: inline-block;
        padding: 2px 8px;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .domain-factual { background: #1f6feb33; color: #58a6ff; }
    .domain-formatting { background: #238636; color: #3fb950; }
    .domain-accuracy { background: #da3633; color: #f85149; }
    .domain-completeness { background: #d29922; color: #e3b341; }
    .domain-general { background: #30363d; color: #8b949e; }
    .header-gradient {
        background: linear-gradient(90deg, #58a6ff 0%, #3fb950 50%, #d29922 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.5rem;
        font-weight: 800;
    }
</style>
""", unsafe_allow_html=True)

# ─── Database Path ────────────────────────────────────────────────────────
DB_PATH = str(PROJECT_DIR / "self_improving_agent.db")


# ─── Helper Functions ─────────────────────────────────────────────────────

def get_agent() -> SelfImprovingAgent:
    """Get or create the agent instance."""
    if "agent" not in st.session_state:
        st.session_state.agent = SelfImprovingAgent(db_path=DB_PATH)
    return st.session_state.agent


def render_metric_card(label: str, value: str, delta: str = ""):
    """Render a styled metric card."""
    delta_html = f'<div style="color: #3fb950; font-size: 0.8rem;">{delta}</div>' if delta else ""
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{label}</div>
        {delta_html}
    </div>
    """, unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="header-gradient">🧠 Self-Improving Agent</div>', unsafe_allow_html=True)
    st.caption("Procedural Memory & Learning Dashboard")
    st.divider()

    page = st.radio(
        "Navigate",
        ["💬 Chat & Correct", "📚 Rule Book", "📈 Learning Curve", "🔬 Run Evaluation", "📊 Before/After"],
        index=0,
    )

    st.divider()
    agent = get_agent()
    state = agent.get_state()

    st.markdown("### Agent Status")
    st.metric("Interactions", state["total_interactions"])
    st.metric("Rules Learned", state["total_rules"])
    st.metric("Corrections", state["total_corrections"])
    st.metric("Current Accuracy", f"{state['accuracy']:.0%}")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 1: CHAT & CORRECT
# ═══════════════════════════════════════════════════════════════════════════

if page == "💬 Chat & Correct":
    st.title("💬 Chat with the Self-Improving Agent")
    st.caption("Ask questions and correct the agent's mistakes. Each correction becomes a learned rule.")

    # Chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "last_response" not in st.session_state:
        st.session_state.last_response = ""
    if "last_question" not in st.session_state:
        st.session_state.last_question = ""

    # Display chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            if msg.get("rules_applied"):
                st.caption(f"📋 Applied {msg['rules_applied']} rules")

    # Chat input
    if prompt := st.chat_input("Ask a question..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        agent = get_agent()
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                response, applied_rules = agent.respond(prompt)
            st.write(response)
            if applied_rules:
                st.caption(f"📋 Applied {len(applied_rules)} learned rules")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "rules_applied": len(applied_rules),
        })
        st.session_state.last_response = response
        st.session_state.last_question = prompt

    # Correction interface
    st.divider()
    st.subheader("✏️ Correction Interface")

    if st.session_state.last_response:
        with st.expander("Correct the last response", expanded=False):
            st.info(f"**Last response:** {st.session_state.last_response[:300]}...")
            correction = st.text_area(
                "Your correction (explain what was wrong and how to fix it):",
                key="correction_input",
                placeholder="e.g., 'The capital of Australia is Canberra, not Sydney. Be careful with commonly confused capitals.'",
            )
            if st.button("Submit Correction", type="primary"):
                if correction:
                    agent = get_agent()
                    with st.spinner("Extracting correction rule..."):
                        rule = agent.handle_correction(
                            st.session_state.last_question,
                            st.session_state.last_response,
                            correction,
                        )
                    st.success(f"✅ Rule extracted and stored!")
                    st.info(f"**Rule:** {rule.rule_text}")
                    st.caption(f"Domain: {rule.domain.value} | Confidence: {rule.confidence:.2f}")
                else:
                    st.warning("Please enter a correction.")
    else:
        st.caption("Send a message first, then you can correct the response.")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 2: RULE BOOK
# ═══════════════════════════════════════════════════════════════════════════

elif page == "📚 Rule Book":
    st.title("📚 Learned Rule Book")
    st.caption("All rules the agent has learned from corrections. These are applied to future responses.")

    memory = ProceduralMemoryStore(db_path=DB_PATH)
    rules = memory.get_all_rules(active_only=False)

    if rules:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        active_rules = [r for r in rules if r.is_active]
        with col1:
            render_metric_card("Active Rules", str(len(active_rules)))
        with col2:
            render_metric_card("Inactive Rules", str(len(rules) - len(active_rules)))
        with col3:
            avg_confidence = sum(r.confidence for r in active_rules) / len(active_rules) if active_rules else 0
            render_metric_card("Avg Confidence", f"{avg_confidence:.2f}")
        with col4:
            total_applications = sum(r.application_count for r in active_rules)
            render_metric_card("Total Applications", str(total_applications))

        st.divider()

        # Domain filter
        domains = list(set(r.domain.value for r in rules))
        selected_domain = st.selectbox("Filter by domain", ["All"] + sorted(domains))

        # Display rules
        for rule in rules:
            if selected_domain != "All" and rule.domain.value != selected_domain:
                continue

            domain_class = f"domain-{rule.domain.value}" if rule.domain.value in ["factual", "formatting", "accuracy", "completeness"] else "domain-general"
            status = "🟢 Active" if rule.is_active else "🔴 Inactive"

            with st.container():
                st.markdown(f"""
                <div class="rule-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <span class="rule-domain {domain_class}">{rule.domain.value.upper()}</span>
                        <span style="color: #8b949e; font-size: 0.8rem;">{status} | Confidence: {rule.confidence:.2f} | Applied: {rule.application_count}x</span>
                    </div>
                    <div style="color: #c9d1d9; font-size: 0.95rem; margin-bottom: 8px;"><strong>Rule:</strong> {rule.rule_text}</div>
                    <div style="color: #8b949e; font-size: 0.8rem;"><strong>Original mistake:</strong> {rule.original_mistake[:150]}...</div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No rules learned yet. Start chatting and correcting the agent to build the rule book!")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 3: LEARNING CURVE
# ═══════════════════════════════════════════════════════════════════════════

elif page == "📈 Learning Curve":
    st.title("📈 Learning Curve")
    st.caption("Visualisation of the agent's improvement over time.")

    memory = ProceduralMemoryStore(db_path=DB_PATH)
    curve = compute_learning_curve(memory)

    if curve:
        # Generate chart
        chart_path = str(PROJECT_DIR / "learning_curve.png")
        generate_learning_curve_chart(memory, output_path=chart_path)

        if os.path.exists(chart_path):
            st.image(chart_path, caption="Agent Learning Curve", use_container_width=True)

        # Data table
        st.divider()
        st.subheader("Raw Data")

        import pandas as pd
        df = pd.DataFrame([{
            "Interaction": p.interaction_number,
            "Cumulative Accuracy": f"{p.cumulative_accuracy:.1%}",
            "Rolling Accuracy (5)": f"{p.rolling_accuracy:.1%}",
            "Error Rate": f"{p.error_rate:.1%}",
            "Total Rules": p.total_rules,
        } for p in curve])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No performance data yet. Interact with the agent or run the evaluation to generate data.")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 4: RUN EVALUATION
# ═══════════════════════════════════════════════════════════════════════════

elif page == "🔬 Run Evaluation":
    st.title("🔬 20-Interaction Evaluation")
    st.caption("Run the full evaluation demonstration: 20 interactions with predefined corrections.")

    st.warning("⚠️ This will run 20 LLM calls with corrections. It takes approximately 3-5 minutes.")

    if st.button("🚀 Run Full Evaluation", type="primary"):
        eval_db = str(PROJECT_DIR / "self_improving_agent_eval.db")

        with st.spinner("Running 20-interaction evaluation..."):
            progress = st.progress(0, text="Starting evaluation...")
            status_text = st.empty()

            # Run evaluation
            results = run_evaluation(db_path=eval_db)

            progress.progress(100, text="Evaluation complete!")

        # Display results
        st.success("✅ Evaluation complete!")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            render_metric_card("Total Interactions", "20")
        with col2:
            render_metric_card("Corrections", str(results["corrections_applied"]))
        with col3:
            render_metric_card("Rules Learned", str(results["total_rules"]))
        with col4:
            render_metric_card("Improvement", f"{results['improvement']:.0%}")

        st.divider()

        # Error rate comparison
        st.subheader("Error Rate Comparison")
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("Early Error Rate (1-5)", f"{results['early_error_rate']:.0%}",
                       help="Error rate in the first 5 interactions")
        with col_b:
            st.metric("Late Error Rate (16-20)", f"{results['late_error_rate']:.0%}",
                       delta=f"-{results['improvement']:.0%}",
                       help="Error rate in the last 5 interactions")

        # Chart
        chart_path = str(PROJECT_DIR / "learning_curve.png")
        eval_memory = ProceduralMemoryStore(db_path=eval_db)
        generate_learning_curve_chart(eval_memory, output_path=chart_path)
        if os.path.exists(chart_path):
            st.image(chart_path, caption="Evaluation Learning Curve", use_container_width=True)

        # Save report
        report_path = str(PROJECT_DIR / "evaluation_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, default=str)
        st.caption(f"Report saved to {report_path}")

    # Show previous results if available
    report_path = str(PROJECT_DIR / "evaluation_report.json")
    if os.path.exists(report_path):
        st.divider()
        st.subheader("Previous Evaluation Results")
        with open(report_path, "r", encoding="utf-8") as f:
            prev_results = json.load(f)

        for interaction in prev_results.get("interactions", []):
            icon = "❌" if interaction.get("was_corrected") else "✅"
            rules_applied = interaction.get("rules_applied", 0)
            rules_text = f" | 📋 {rules_applied} rules" if rules_applied else ""
            with st.expander(f"{icon} Interaction {interaction['number']}: {interaction['question'][:60]}...{rules_text}"):
                st.write(f"**Response:** {interaction['response']}")
                if interaction.get("was_corrected"):
                    st.error(f"**Rule extracted:** {interaction.get('rule_extracted', 'N/A')}")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 5: BEFORE/AFTER COMPARISON
# ═══════════════════════════════════════════════════════════════════════════

elif page == "📊 Before/After":
    st.title("📊 Before/After Comparison")
    st.caption("Compare agent behaviour before and after learning from corrections.")

    memory = ProceduralMemoryStore(db_path=DB_PATH)
    rules = memory.get_all_rules()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("🚫 Before (0 Rules)")
        st.markdown("""
        <div class="rule-card">
            <p style="color: #f85149;">The agent has NO procedural memory.</p>
            <ul style="color: #8b949e;">
                <li>Makes the same mistakes repeatedly</li>
                <li>No awareness of past corrections</li>
                <li>Generic responses with no learned patterns</li>
                <li>No improvement over time</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.subheader(f"✅ After ({len(rules)} Rules)")
        st.markdown(f"""
        <div class="rule-card">
            <p style="color: #3fb950;">The agent has learned {len(rules)} correction rules.</p>
            <ul style="color: #c9d1d9;">
                <li>Applies learned rules to new questions</li>
                <li>Avoids previously corrected mistakes</li>
                <li>Responses augmented with procedural memory</li>
                <li>Error rate decreases measurably</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.divider()

    # Live comparison test
    st.subheader("🔄 Live Comparison")
    test_question = st.text_input(
        "Enter a question to compare responses:",
        placeholder="e.g., What is the capital of Australia?",
    )

    if test_question and st.button("Compare", type="primary"):
        agent = get_agent()

        col_before, col_after = st.columns(2)

        with col_before:
            st.markdown("### Without Rules")
            from agent import llm_call
            with st.spinner("Generating..."):
                base_response = llm_call(
                    "You are a helpful AI assistant. Answer the question clearly and accurately.",
                    test_question,
                )
            st.write(base_response)

        with col_after:
            st.markdown("### With Learned Rules")
            with st.spinner("Generating with rules..."):
                augmented_response, applied = agent.respond(test_question)
            st.write(augmented_response)
            if applied:
                st.success(f"Applied {len(applied)} rules")
