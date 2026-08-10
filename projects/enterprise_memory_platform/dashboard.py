"""
Enterprise AI Memory Platform — Streamlit Admin Dashboard
============================================================
Rich admin dashboard showing:
  - Platform overview with live stats for all tenants
  - Per-tenant memory inspection (all 4 types)
  - Knowledge graph visualisation
  - Consolidation management
  - Multi-tenant isolation verification
  - Demo data seeding

Run:
    streamlit run projects/enterprise_memory_platform/dashboard.py
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime, timezone

# Path setup
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

import streamlit as st

from models import (
    EpisodicMemoryCreate, EpisodicQueryRequest,
    SemanticMemoryCreate, SemanticQueryRequest,
    ProceduralRuleCreate, ProceduralQueryRequest,
    EntityCreate, RelationshipCreate, GraphQueryRequest,
    RuleDomain, ConsolidationConfig,
)
from router import MemoryRouter
from consolidation import ConsolidationWorker


# ─── Page Config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Enterprise AI Memory Platform — Admin",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Premium CSS ──────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    .stApp { background-color: #0a0a1a; font-family: 'Inter', sans-serif; }

    .platform-header {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 16px;
        padding: 32px;
        margin-bottom: 24px;
        text-align: center;
    }
    .platform-title {
        font-size: 2.4rem;
        font-weight: 800;
        background: linear-gradient(90deg, #667eea, #764ba2, #f093fb);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 8px;
    }
    .platform-subtitle { color: #8b8fad; font-size: 1rem; }

    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid rgba(102, 126, 234, 0.2);
        border-radius: 14px;
        padding: 24px;
        text-align: center;
        transition: all 0.3s ease;
    }
    .metric-card:hover { border-color: rgba(102, 126, 234, 0.5); transform: translateY(-2px); }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-label { color: #8b8fad; font-size: 0.85rem; margin-top: 4px; font-weight: 500; }

    .tenant-card {
        background: #12121f;
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 12px;
        transition: all 0.2s;
    }
    .tenant-card:hover { border-color: rgba(102, 126, 234, 0.3); }
    .tenant-name { font-size: 1.2rem; font-weight: 700; color: #e0e0ff; }
    .tenant-id { color: #667eea; font-size: 0.8rem; font-weight: 500; }

    .memory-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-right: 6px;
    }
    .badge-episodic { background: rgba(52, 152, 219, 0.15); color: #3498db; }
    .badge-semantic { background: rgba(46, 204, 113, 0.15); color: #2ecc71; }
    .badge-procedural { background: rgba(231, 76, 60, 0.15); color: #e74c3c; }
    .badge-graph { background: rgba(241, 196, 15, 0.15); color: #f1c40f; }

    .isolation-pass { background: rgba(46, 204, 113, 0.1); border: 1px solid #2ecc71; border-radius: 12px; padding: 16px; }
    .isolation-fail { background: rgba(231, 76, 60, 0.1); border: 1px solid #e74c3c; border-radius: 12px; padding: 16px; }
</style>
""", unsafe_allow_html=True)


# ─── State Initialisation ─────────────────────────────────────────────────
DATA_DIR = str(PROJECT_DIR / "data")

@st.cache_resource
def get_router():
    return MemoryRouter(data_dir=DATA_DIR)

@st.cache_resource
def get_worker():
    return ConsolidationWorker(get_router())


def render_metric(label: str, value: str | int, color: str = "#667eea"):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value" style="background: linear-gradient(135deg, {color}, #764ba2); -webkit-background-clip: text; -webkit-text-fill-color: transparent;">{value}</div>
        <div class="metric-label">{label}</div>
    </div>
    """, unsafe_allow_html=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align: center; padding: 12px 0;">
        <div style="font-size: 2.2rem;">🧠</div>
        <div style="font-size: 1.1rem; font-weight: 700; color: #e0e0ff;">Memory Platform</div>
        <div style="color: #667eea; font-size: 0.8rem;">Admin Dashboard v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    page = st.radio(
        "Navigate",
        [
            "🏠 Platform Overview",
            "🔍 Memory Inspector",
            "🕸️ Knowledge Graph",
            "⚙️ Consolidation",
            "🔒 Isolation Check",
            "📦 Seed Demo Data",
        ],
        index=0,
    )

    st.divider()

    # Quick stats
    router = get_router()
    tenants = router.tenants.list_tenants()
    st.caption(f"**Tenants:** {len(tenants)}")
    for t in tenants[:5]:
        st.caption(f"  • {t.name}")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 1: PLATFORM OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

if page == "🏠 Platform Overview":
    st.markdown("""
    <div class="platform-header">
        <div class="platform-title">Enterprise AI Memory Platform</div>
        <div class="platform-subtitle">Memory-as-a-Service for AI Agents — 4 Memory Types, Multi-Tenant Isolation</div>
    </div>
    """, unsafe_allow_html=True)

    router = get_router()
    stats = router.get_platform_stats()

    # Top-level metrics
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1: render_metric("Active Tenants", stats["active_tenants"], "#667eea")
    with c2: render_metric("Total Memories", stats["total_memories"], "#2ecc71")
    with c3: render_metric("Procedural Rules", stats["total_rules"], "#e74c3c")
    with c4: render_metric("Graph Entities", stats["total_graph_entities"], "#f1c40f")
    with c5: render_metric("Total Tenants", stats["total_tenants"], "#3498db")

    st.divider()

    # Per-tenant cards
    st.subheader("Tenant Overview")

    for ts in stats.get("tenants", []):
        with st.container():
            st.markdown(f"""
            <div class="tenant-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div class="tenant-name">{ts.get('tenant_name', ts['tenant_id'])}</div>
                        <div class="tenant-id">@{ts['tenant_id']}</div>
                    </div>
                    <div style="text-align: right;">
                        <span class="memory-badge badge-episodic">Episodic: {ts['episodic_count']}</span>
                        <span class="memory-badge badge-semantic">Semantic: {ts['semantic_count']}</span>
                        <span class="memory-badge badge-procedural">Procedural: {ts['procedural_count']}</span>
                        <span class="memory-badge badge-graph">Graph: {ts['graph_entities']}E / {ts['graph_relationships']}R</span>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 2: MEMORY INSPECTOR
# ═══════════════════════════════════════════════════════════════════════════

elif page == "🔍 Memory Inspector":
    st.title("🔍 Memory Inspector")
    st.caption("Browse and search memories across all 4 types for any tenant.")

    router = get_router()
    tenants = router.tenants.list_tenants()
    tenant_options = {t.name: t.tenant_id for t in tenants}

    if not tenant_options:
        st.info("No tenants available. Seed demo data first.")
    else:
        selected_name = st.selectbox("Select Tenant", list(tenant_options.keys()))
        selected_tid = tenant_options[selected_name]

        tab_ep, tab_sem, tab_proc, tab_graph = st.tabs([
            "📝 Episodic", "🧠 Semantic", "📋 Procedural", "🕸️ Knowledge Graph"
        ])

        # Episodic tab
        with tab_ep:
            st.subheader(f"Episodic Memories — {selected_name}")
            episodes = router.episodic.query(selected_tid, EpisodicQueryRequest(limit=50))
            if episodes:
                import pandas as pd
                df = pd.DataFrame([{
                    "ID": ep.memory_id,
                    "Session": ep.session_id,
                    "Role": ep.role,
                    "Content": ep.content[:100],
                    "Importance": f"{ep.importance_score:.2f}",
                    "Consolidated": "✅" if ep.is_consolidated else "❌",
                    "Time": ep.timestamp[:19],
                } for ep in episodes])
                st.dataframe(df, use_container_width=True)
            else:
                st.info("No episodic memories.")

        # Semantic tab
        with tab_sem:
            st.subheader(f"Semantic Memories — {selected_name}")
            all_semantic = router.semantic.get_all(selected_tid)
            if all_semantic:
                import pandas as pd
                df = pd.DataFrame([{
                    "ID": sm.memory_id,
                    "Fact": sm.fact[:80],
                    "Category": sm.category,
                    "Confidence": f"{sm.confidence:.2f}",
                } for sm in all_semantic])
                st.dataframe(df, use_container_width=True)

                # Search
                query = st.text_input("Search semantic memories:", key="sem_search")
                if query:
                    results = router.semantic.query(selected_tid, SemanticQueryRequest(query=query, limit=5))
                    for r in results:
                        st.markdown(f"**{r.fact}** (relevance: {r.metadata.get('relevance_score', 'N/A')})")
            else:
                st.info("No semantic memories.")

        # Procedural tab
        with tab_proc:
            st.subheader(f"Procedural Rules — {selected_name}")
            rules = router.procedural.get_all(selected_tid, active_only=False)
            if rules:
                for rule in rules:
                    status_icon = "🟢" if rule.is_active else "🔴"
                    with st.expander(f"{status_icon} [{rule.domain.value.upper()}] {rule.rule_text[:80]}..."):
                        st.write(f"**Rule:** {rule.rule_text}")
                        st.write(f"**Original mistake:** {rule.original_mistake[:200]}")
                        st.write(f"**Correction:** {rule.correction[:200]}")
                        c1, c2, c3 = st.columns(3)
                        c1.metric("Confidence", f"{rule.confidence:.2f}")
                        c2.metric("Applications", rule.application_count)
                        c3.metric("Active", "Yes" if rule.is_active else "No")
            else:
                st.info("No procedural rules.")

        # Graph tab
        with tab_graph:
            st.subheader(f"Knowledge Graph — {selected_name}")
            graph_stats = router.graph.get_stats(selected_tid)
            c1, c2 = st.columns(2)
            c1.metric("Entities", graph_stats.total_entities)
            c2.metric("Relationships", graph_stats.total_relationships)

            entities = router.graph.get_all_entities(selected_tid)
            if entities:
                import pandas as pd
                df = pd.DataFrame([{
                    "Name": e.name,
                    "Type": e.entity_type,
                    "Connections": e.connections,
                    "Description": e.description[:60],
                } for e in entities])
                st.dataframe(df, use_container_width=True)

            rels = router.graph.get_all_relationships(selected_tid)
            if rels:
                st.subheader("Relationships")
                import pandas as pd
                df = pd.DataFrame([{
                    "Source": r.source,
                    "Relation": r.relation_type,
                    "Target": r.target,
                    "Confidence": f"{r.confidence:.2f}",
                } for r in rels])
                st.dataframe(df, use_container_width=True)


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 3: KNOWLEDGE GRAPH VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════

elif page == "🕸️ Knowledge Graph":
    st.title("🕸️ Knowledge Graph Visualisation")

    router = get_router()
    tenants = router.tenants.list_tenants()
    tenant_options = {t.name: t.tenant_id for t in tenants}

    if not tenant_options:
        st.info("No tenants. Seed demo data first.")
    else:
        selected_name = st.selectbox("Select Tenant", list(tenant_options.keys()), key="graph_tenant")
        selected_tid = tenant_options[selected_name]

        stats = router.graph.get_stats(selected_tid)
        c1, c2 = st.columns(2)
        c1.metric("Entities", stats.total_entities)
        c2.metric("Relationships", stats.total_relationships)

        if stats.total_entities > 0:
            # Generate pyvis visualisation
            try:
                from pyvis.network import Network
                import tempfile

                graph = router.graph._get_graph(selected_tid)

                color_map = {
                    "person": "#e74c3c", "company": "#3498db", "technology": "#2ecc71",
                    "concept": "#f39c12", "place": "#9b59b6", "product": "#1abc9c",
                    "unknown": "#95a5a6",
                }

                net = Network(height="600px", width="100%", bgcolor="#0a0a1a",
                              font_color="white", directed=True, notebook=False)

                for node, data in graph.nodes(data=True):
                    etype = data.get("entity_type", "unknown")
                    color = color_map.get(etype, "#95a5a6")
                    degree = graph.degree(node)
                    size = max(15, min(50, 10 + degree * 5))
                    title = f"Type: {etype}\n{data.get('description', '')}"
                    net.add_node(node, label=node, color=color, size=size, title=title)

                for src, tgt, data in graph.edges(data=True):
                    rel = data.get("relation_type", "related_to")
                    conf = data.get("confidence", 0.8)
                    net.add_edge(src, tgt, label=rel, title=f"{rel} ({conf:.2f})",
                                 color="#444", width=max(1, conf * 3))

                net.set_options('{"physics": {"forceAtlas2Based": {"gravitationalConstant": -50}, "solver": "forceAtlas2Based"}}')

                with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w", encoding="utf-8") as f:
                    net.save_graph(f.name)
                    html_path = f.name

                with open(html_path, "r", encoding="utf-8") as f:
                    html_content = f.read()

                st.components.v1.html(html_content, height=650, scrolling=True)

            except ImportError:
                st.warning("pyvis not installed. Run: `pip install pyvis`")
        else:
            st.info("No entities in this tenant's graph. Seed demo data first.")

        # Path finder
        st.divider()
        st.subheader("🔎 Path Finder")
        c1, c2 = st.columns(2)
        with c1:
            source_entity = st.text_input("Source entity:", key="path_src")
        with c2:
            target_entity = st.text_input("Target entity:", key="path_tgt")

        if source_entity and target_entity and st.button("Find Path"):
            result = router.find_path(selected_tid, source_entity, target_entity)
            if result.data:
                for step in result.data:
                    st.markdown(f"**{step['from']}** →[{step['relation']}]→ **{step['to']}**")
            else:
                st.warning("No path found between these entities.")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 4: CONSOLIDATION
# ═══════════════════════════════════════════════════════════════════════════

elif page == "⚙️ Consolidation":
    st.title("⚙️ Consolidation Manager")
    st.caption("Manage memory consolidation: compress old episodes, promote rules, prune low-importance memories.")

    router = get_router()
    worker = get_worker()
    tenants = router.tenants.list_tenants()

    # Configuration
    with st.expander("⚙️ Consolidation Configuration"):
        c1, c2 = st.columns(2)
        with c1:
            threshold = st.number_input("Episode threshold", value=100, min_value=10)
            batch_size = st.number_input("Batch size", value=50, min_value=5)
        with c2:
            min_importance = st.slider("Min importance to keep", 0.0, 1.0, 0.3)
            rule_promotion = st.slider("Rule promotion threshold", 0.0, 1.0, 0.85)

        worker.config = ConsolidationConfig(
            episode_threshold=threshold,
            batch_size=batch_size,
            min_importance_to_keep=min_importance,
            rule_confidence_promotion=rule_promotion,
        )

    st.divider()

    # Per-tenant controls
    for t in tenants:
        col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
        with col1:
            st.markdown(f"**{t.name}** (`{t.tenant_id}`)")
        with col2:
            ep_count = router.episodic.count_unconsolidated(t.tenant_id)
            st.caption(f"Pending: {ep_count}")
        with col3:
            st.caption(f"Rules: {router.procedural.count(t.tenant_id)}")
        with col4:
            if st.button("Run", key=f"consol_{t.tenant_id}"):
                with st.spinner(f"Consolidating {t.name}..."):
                    record = worker.force_consolidation(t.tenant_id)
                st.success(f"✅ {record.summary_text}")

    st.divider()

    # Run all
    if st.button("🔄 Run Consolidation for All Tenants", type="primary"):
        with st.spinner("Running consolidation for all tenants..."):
            records = worker.run_all_tenants()
        for r in records:
            icon = "✅" if r.status.value == "completed" else "❌"
            st.markdown(f"{icon} **{r.tenant_id}**: {r.summary_text}")

    # History
    st.divider()
    st.subheader("📜 Consolidation History")
    history = worker.get_history()
    if history:
        import pandas as pd
        df = pd.DataFrame([{
            "Tenant": r.tenant_id,
            "Status": r.status.value,
            "Episodes": f"{r.episodes_consolidated}/{r.episodes_processed}",
            "Pruned": r.memories_pruned,
            "Rules Promoted": r.rules_promoted,
            "Time": r.timestamp[:19],
        } for r in history])
        st.dataframe(df, use_container_width=True)
    else:
        st.info("No consolidation runs yet.")


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 5: ISOLATION CHECK
# ═══════════════════════════════════════════════════════════════════════════

elif page == "🔒 Isolation Check":
    st.title("🔒 Multi-Tenant Isolation Verification")
    st.caption("Verify that tenant A cannot read tenant B's memories.")

    if st.button("🔍 Run Isolation Test", type="primary"):
        router = get_router()

        # Seed test data for acme_corp
        router.tenants.create_tenant("acme_corp", "Acme Corporation")
        router.tenants.create_tenant("globex_inc", "Globex Inc.")

        router.store_episodic("acme_corp", EpisodicMemoryCreate(
            session_id="isolation-test",
            content="CONFIDENTIAL: Acme's secret product launch date is January 15",
            importance_score=0.95,
        ))
        router.store_semantic("acme_corp", SemanticMemoryCreate(
            fact="Acme Corporation's quarterly revenue is $50 million",
            category="financial",
        ))
        router.store_procedural("acme_corp", ProceduralRuleCreate(
            original_mistake="Disclosed internal pricing",
            correction="Never share Acme's internal pricing with external parties",
            rule_text="All pricing information is confidential to Acme Corporation",
            domain=RuleDomain.ACCURACY,
        ))

        # Cross-check: query as globex
        ep = router.episodic.query("globex_inc", EpisodicQueryRequest(limit=100))
        sem_results = router.semantic.get_all("globex_inc")
        proc = router.procedural.get_all("globex_inc")

        acme_leak = False
        for e in ep:
            if "acme" in e.content.lower() or "confidential" in e.content.lower():
                acme_leak = True
        for s in sem_results:
            if "acme" in s.fact.lower():
                acme_leak = True
        for r in proc:
            if "acme" in r.rule_text.lower():
                acme_leak = True

        if not acme_leak:
            st.markdown("""
            <div class="isolation-pass">
                <h3 style="color: #2ecc71;">✅ ISOLATION VERIFIED</h3>
                <p style="color: #c9d1d9;">
                    Globex Inc. cannot access any of Acme Corporation's memories.
                    Episodic, semantic, and procedural stores are fully isolated.
                </p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="isolation-fail">
                <h3 style="color: #e74c3c;">❌ ISOLATION BREACH DETECTED</h3>
                <p style="color: #c9d1d9;">Cross-tenant data leakage was detected.</p>
            </div>
            """, unsafe_allow_html=True)

        # Show counts
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.subheader("Acme Corp (data owner)")
            st.metric("Episodic", router.episodic.count("acme_corp"))
            st.metric("Semantic", router.semantic.count("acme_corp"))
            st.metric("Procedural", router.procedural.count("acme_corp"))
        with c2:
            st.subheader("Globex Inc (should see nothing)")
            st.metric("Episodic (from Acme)", len([e for e in ep if "acme" in e.content.lower()]))
            st.metric("Semantic (from Acme)", len([s for s in sem_results if "acme" in s.fact.lower()]))
            st.metric("Procedural (from Acme)", len([r for r in proc if "acme" in r.rule_text.lower()]))


# ═══════════════════════════════════════════════════════════════════════════
#  PAGE 6: SEED DEMO DATA
# ═══════════════════════════════════════════════════════════════════════════

elif page == "📦 Seed Demo Data":
    st.title("📦 Seed Demo Data")
    st.caption("Populate 3 demo tenants with realistic memory data for demonstration.")

    if st.button("🌱 Seed All Demo Data", type="primary"):
        router = get_router()

        with st.spinner("Seeding demo data..."):
            # ── TENANT 1: Acme Corp ──
            router.tenants.create_tenant("acme_corp", "Acme Corporation")

            acme_episodes = [
                ("session-1", "user", "What were our Q3 revenue numbers?", 0.8),
                ("session-1", "assistant", "Acme's Q3 2025 revenue was $47.2M, up 12% from Q3 2024.", 0.8),
                ("session-1", "user", "How about customer churn?", 0.6),
                ("session-1", "assistant", "Customer churn was 4.2% in Q3, down from 5.1% in Q2.", 0.6),
                ("session-2", "user", "Draft a message to the engineering team about the API migration", 0.7),
                ("session-2", "assistant", "Subject: API Migration to v3 — Timeline and Action Items...", 0.7),
                ("session-3", "user", "What are the key risks for next quarter?", 0.9),
                ("session-3", "assistant", "Top risks: 1) Supply chain delays, 2) New EU regulations, 3) Competitor launch.", 0.9),
            ]
            for sid, role, content, imp in acme_episodes:
                router.store_episodic("acme_corp", EpisodicMemoryCreate(
                    session_id=sid, content=content, role=role, importance_score=imp,
                ))

            acme_facts = [
                ("Acme's CEO is Jane Smith", "profile", 0.95),
                ("Acme prefers concise, bullet-point reports", "preference", 0.9),
                ("Acme's fiscal year ends in March", "fact", 0.85),
                ("Acme's main competitor is Globex Inc", "knowledge", 0.8),
            ]
            for fact, cat, conf in acme_facts:
                router.store_semantic("acme_corp", SemanticMemoryCreate(fact=fact, category=cat, confidence=conf))

            router.store_procedural("acme_corp", ProceduralRuleCreate(
                original_mistake="Used informal tone in executive summary",
                correction="Use formal business language for all executive communications",
                rule_text="Always use formal tone when drafting executive-level documents for Acme",
                domain=RuleDomain.TONE, confidence=0.9,
            ))

            # Acme graph
            for name, etype in [("Acme Corporation", "company"), ("Jane Smith", "person"),
                                 ("API v3", "technology"), ("Globex Inc", "company"), ("EU Regulations", "concept")]:
                router.add_entity("acme_corp", EntityCreate(name=name, entity_type=etype))
            for src, tgt, rel in [("Jane Smith", "Acme Corporation", "ceo_of"),
                                   ("Acme Corporation", "API v3", "developing"),
                                   ("Acme Corporation", "Globex Inc", "competes_with")]:
                router.add_relationship("acme_corp", RelationshipCreate(source=src, target=tgt, relation_type=rel))

            # ── TENANT 2: Globex Inc ──
            router.tenants.create_tenant("globex_inc", "Globex Inc.")

            globex_episodes = [
                ("session-1", "user", "Summarise the latest product roadmap", 0.7),
                ("session-1", "assistant", "Globex Q4 roadmap: 1) Launch Widget Pro, 2) Expand APAC, 3) AI assistant beta.", 0.7),
                ("session-2", "user", "What's our hiring plan for engineering?", 0.8),
                ("session-2", "assistant", "Plan: 15 senior engineers, 10 junior, 5 ML specialists by Q1 2026.", 0.8),
            ]
            for sid, role, content, imp in globex_episodes:
                router.store_episodic("globex_inc", EpisodicMemoryCreate(
                    session_id=sid, content=content, role=role, importance_score=imp,
                ))

            globex_facts = [
                ("Globex CEO is Hank Scorpio", "profile", 0.95),
                ("Globex uses metric system for all reports", "preference", 0.85),
                ("Globex annual revenue is $120M", "fact", 0.9),
            ]
            for fact, cat, conf in globex_facts:
                router.store_semantic("globex_inc", SemanticMemoryCreate(fact=fact, category=cat, confidence=conf))

            for name, etype in [("Globex Inc", "company"), ("Hank Scorpio", "person"),
                                 ("Widget Pro", "product"), ("APAC Region", "place")]:
                router.add_entity("globex_inc", EntityCreate(name=name, entity_type=etype))
            for src, tgt, rel in [("Hank Scorpio", "Globex Inc", "ceo_of"),
                                   ("Globex Inc", "Widget Pro", "developing")]:
                router.add_relationship("globex_inc", RelationshipCreate(source=src, target=tgt, relation_type=rel))

            # ── TENANT 3: Initech ──
            router.tenants.create_tenant("initech", "Initech")

            initech_episodes = [
                ("session-1", "user", "What's the status of the TPS report system?", 0.6),
                ("session-1", "assistant", "TPS report migration is 75% complete. Deadline: end of month.", 0.6),
                ("session-2", "user", "Review the latest compliance audit findings", 0.9),
                ("session-2", "assistant", "Audit found 3 minor issues: 1) Missing access logs, 2) Outdated certs, 3) Backup gaps.", 0.9),
            ]
            for sid, role, content, imp in initech_episodes:
                router.store_episodic("initech", EpisodicMemoryCreate(
                    session_id=sid, content=content, role=role, importance_score=imp,
                ))

            initech_facts = [
                ("Initech specialises in enterprise software consulting", "fact", 0.9),
                ("Initech requires all reports to include TPS cover sheets", "preference", 0.95),
            ]
            for fact, cat, conf in initech_facts:
                router.store_semantic("initech", SemanticMemoryCreate(fact=fact, category=cat, confidence=conf))

            router.store_procedural("initech", ProceduralRuleCreate(
                original_mistake="Forgot TPS report cover sheet",
                correction="Always include the TPS cover sheet on every report",
                rule_text="Every report for Initech MUST include the TPS cover sheet as the first page",
                domain=RuleDomain.FORMATTING, confidence=0.95,
            ))

        st.success("✅ Demo data seeded for 3 tenants: Acme Corporation, Globex Inc., and Initech")
        st.rerun()
