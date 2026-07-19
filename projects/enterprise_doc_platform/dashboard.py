"""Enterprise Document Intelligence Platform — Streamlit Admin Dashboard.

Run with: streamlit run dashboard.py
"""

import json
import time

import httpx
import streamlit as st

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
API_BASE = "http://localhost:8000/api/v1"

st.set_page_config(
    page_title="📊 Enterprise Doc Intelligence — Admin",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-value {
        font-size: 2.5rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.8;
    }
    .stButton>button {
        border-radius: 8px;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# API Helpers
# ---------------------------------------------------------------------------
def api_get(endpoint: str) -> dict | list | None:
    """GET request to the API."""
    try:
        r = httpx.get(f"{API_BASE}{endpoint}", timeout=30)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"API Error: {e}")
        return None


def api_post(endpoint: str, data: dict | None = None, files: dict | None = None) -> dict | None:
    """POST request to the API."""
    try:
        if files:
            r = httpx.post(f"{API_BASE}{endpoint}", files=files, timeout=60)
        else:
            r = httpx.post(f"{API_BASE}{endpoint}", json=data, timeout=60)
        r.raise_for_status()
        return r.json()
    except httpx.HTTPStatusError as e:
        st.error(f"API Error ({e.response.status_code}): {e.response.text}")
        return None
    except Exception as e:
        st.error(f"Connection Error: {e}")
        return None


def api_delete(endpoint: str) -> bool:
    """DELETE request to the API."""
    try:
        r = httpx.delete(f"{API_BASE}{endpoint}", timeout=30)
        r.raise_for_status()
        return True
    except Exception as e:
        st.error(f"API Error: {e}")
        return False


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
st.sidebar.markdown("# 🏢 Admin Dashboard")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigation",
    ["📊 Overview", "🏢 Tenants", "📄 Documents", "🔍 Query", "📈 Evaluation"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.markdown("**Platform Status**")

# Check API health
try:
    health = httpx.get("http://localhost:8000/health", timeout=5).json()
    st.sidebar.success(f"✅ API Online (v{health.get('version', '?')})")
    st.sidebar.caption(f"Uptime: {health.get('uptime_seconds', 0):.0f}s")
except Exception:
    st.sidebar.error("❌ API Offline — Start with `uvicorn app.main:app`")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------
if page == "📊 Overview":
    st.markdown('<div class="main-header">📊 Platform Overview</div>', unsafe_allow_html=True)

    tenants = api_get("/tenants/")
    if tenants:
        col1, col2, col3 = st.columns(3)
        total_docs = sum(t.get("document_count", 0) for t in tenants)
        total_chunks = sum(t.get("chunk_count", 0) for t in tenants)

        with col1:
            st.metric("Total Tenants", len(tenants))
        with col2:
            st.metric("Total Documents", total_docs)
        with col3:
            st.metric("Total Chunks", total_chunks)

        st.markdown("---")
        st.subheader("Tenant Summary")

        for t in tenants:
            with st.expander(f"🏢 {t['name']} ({t['tenant_id']})", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Documents", t.get("document_count", 0))
                c2.metric("Chunks", t.get("chunk_count", 0))
                c3.caption(f"Created: {t.get('created_at', '?')[:10]}")
    else:
        st.info("No tenants registered yet. Go to 'Tenants' to create one.")

elif page == "🏢 Tenants":
    st.markdown('<div class="main-header">🏢 Tenant Management</div>', unsafe_allow_html=True)

    # Create tenant
    with st.form("create_tenant"):
        st.subheader("Create New Tenant")
        name = st.text_input("Tenant Name")
        description = st.text_area("Description")
        submitted = st.form_submit_button("Create Tenant", type="primary")

        if submitted and name:
            result = api_post("/tenants/", {"name": name, "description": description})
            if result:
                st.success(f"✅ Tenant '{result['name']}' created (ID: {result['tenant_id']})")
                st.rerun()

    st.markdown("---")

    # List tenants
    st.subheader("Existing Tenants")
    tenants = api_get("/tenants/")
    if tenants:
        for t in tenants:
            col1, col2 = st.columns([4, 1])
            with col1:
                st.write(f"**{t['name']}** (`{t['tenant_id']}`)")
                st.caption(f"{t.get('description', '')} • {t.get('document_count', 0)} docs • {t.get('chunk_count', 0)} chunks")
            with col2:
                if st.button("🗑️ Delete", key=f"del_{t['tenant_id']}"):
                    if api_delete(f"/tenants/{t['tenant_id']}"):
                        st.success(f"Deleted tenant '{t['tenant_id']}'")
                        st.rerun()

elif page == "📄 Documents":
    st.markdown('<div class="main-header">📄 Document Management</div>', unsafe_allow_html=True)

    tenants = api_get("/tenants/")
    if not tenants:
        st.warning("No tenants available. Create a tenant first.")
    else:
        tenant_options = {t["name"]: t["tenant_id"] for t in tenants}
        selected_tenant_name = st.selectbox("Select Tenant", list(tenant_options.keys()))
        tenant_id = tenant_options[selected_tenant_name]

        # Upload
        st.subheader("Upload Document")
        uploaded_file = st.file_uploader(
            "Choose a file (PDF, TXT, MD)",
            type=["pdf", "txt", "md"],
        )

        if uploaded_file and st.button("📤 Upload & Process", type="primary"):
            with st.spinner("Uploading and processing..."):
                files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
                result = api_post(f"/documents/{tenant_id}/upload", files=files)
                if result:
                    st.success(f"✅ Document uploaded! ID: {result['document_id']} (Status: {result['status']})")
                    st.info("⏳ Processing in background. Refresh to see status update.")

        # List documents
        st.markdown("---")
        st.subheader("Documents")
        docs_data = api_get(f"/documents/{tenant_id}")
        if docs_data and docs_data.get("documents"):
            for doc in docs_data["documents"]:
                status_emoji = {"ready": "✅", "processing": "⏳", "pending": "📋", "failed": "❌"}.get(doc["status"], "❓")
                st.write(f"{status_emoji} **{doc['filename']}** — {doc['status']} | {doc['chunk_count']} chunks | {doc['file_size']} bytes")
        else:
            st.info("No documents uploaded yet.")

elif page == "🔍 Query":
    st.markdown('<div class="main-header">🔍 RAG Query Interface</div>', unsafe_allow_html=True)

    tenants = api_get("/tenants/")
    if not tenants:
        st.warning("No tenants available.")
    else:
        tenant_options = {t["name"]: t["tenant_id"] for t in tenants}
        selected_tenant_name = st.selectbox("Select Tenant", list(tenant_options.keys()))
        tenant_id = tenant_options[selected_tenant_name]

        # Query form
        question = st.text_input("Ask a question", placeholder="What is the main topic discussed in the documents?")

        col1, col2, col3 = st.columns(3)
        with col1:
            top_k = st.slider("Top K results", 1, 10, 5)
        with col2:
            use_hybrid = st.checkbox("Hybrid Search", value=True)
        with col3:
            use_reranking = st.checkbox("Cross-Encoder Re-ranking", value=True)

        if st.button("🔍 Search", type="primary") and question:
            with st.spinner("Processing query..."):
                result = api_post(f"/query/{tenant_id}", {
                    "question": question,
                    "top_k": top_k,
                    "use_hybrid": use_hybrid,
                    "use_reranking": use_reranking,
                })

                if result:
                    # Answer
                    st.markdown("### 💡 Answer")
                    st.markdown(result["answer"])

                    # Metrics
                    col1, col2, col3 = st.columns(3)
                    conf = result.get("confidence_score", 0)
                    conf_color = "green" if conf > 0.7 else "orange" if conf > 0.4 else "red"
                    col1.metric("Confidence", f"{conf:.1%}")
                    col2.metric("Method", result.get("retrieval_method", "?"))
                    col3.metric("Time", f"{result.get('processing_time_ms', 0):.0f}ms")

                    # Sources
                    st.markdown("### 📄 Source Chunks")
                    for i, chunk in enumerate(result.get("source_chunks", []), 1):
                        with st.expander(f"Source {i}: {chunk['source_file']} (Page {chunk['page']}) — Score: {chunk['similarity_score']:.4f}"):
                            st.text(chunk["text"])

elif page == "📈 Evaluation":
    st.markdown('<div class="main-header">📈 RAGAS Evaluation</div>', unsafe_allow_html=True)
    st.info("Run the evaluation script: `python evaluation/evaluate.py`")
    st.markdown("""
    ### Evaluation Metrics
    - **Faithfulness**: Is the answer grounded in the retrieved context?
    - **Answer Relevancy**: Is the answer relevant to the question?
    - **Context Precision**: Are retrieved chunks relevant and well-ordered?

    ### How to Run
    ```bash
    cd projects/enterprise_doc_platform
    python evaluation/evaluate.py
    ```

    The results will be saved to `evaluation/ragas_report.json`.
    """)

    # Try to load existing results
    try:
        with open("evaluation/ragas_report.json") as f:
            report = json.load(f)
        st.subheader("Latest Evaluation Results")
        st.json(report)
    except FileNotFoundError:
        st.info("No evaluation results found yet.")
