# Week 3 — Weekly Assessment
## Embeddings, RAG & Vector Databases

---

### Question 1 (Conceptual)
**What does an embedding represent? Why does cosine similarity work as a measure of semantic closeness?**

An **embedding** is a dense numerical vector representation of text (or any data) in a high-dimensional space, typically ranging from 384 to 1536 dimensions. Each dimension captures a learned semantic feature, so words, sentences, or documents with similar meanings are mapped to nearby points in this vector space. For example, "dog" and "puppy" would have similar embeddings despite being different words, because embedding models learn from large text corpora that these terms appear in similar contexts.

**Cosine similarity** works as a measure of semantic closeness because it measures the **angle** between two vectors rather than their magnitude. This is critical for embeddings because:

1. **Direction encodes meaning**: Embedding models are trained so that the *direction* of a vector captures semantic content. Two sentences about the same topic will point in similar directions.
2. **Magnitude-invariant**: A longer document might produce a vector with a larger magnitude, but cosine similarity normalizes this away — it only cares about the angle.
3. **Bounded output**: Cosine similarity produces values between −1 (opposite meaning) and +1 (identical meaning), making it easy to interpret and threshold.

Mathematically: `cos(θ) = (A · B) / (||A|| × ||B||)`, where the dot product captures alignment and the norms normalize for length.

---

### Question 2 (Conceptual)
**Explain the tradeoffs between chunk size, retrieval accuracy, and context window usage in RAG.**

Chunk size is one of the most impactful hyperparameters in a RAG system. The tradeoffs are:

| Factor | Small Chunks (e.g., 256 tokens) | Large Chunks (e.g., 1024 tokens) |
|--------|--------------------------------|----------------------------------|
| **Retrieval Precision** | ✅ Higher — each chunk is focused on one topic, so the retrieved content is more precisely relevant. | ❌ Lower — chunks may contain mixed topics, diluting relevance. |
| **Retrieval Recall** | ❌ Lower — important context may be split across multiple chunks and some pieces may be missed. | ✅ Higher — more context is captured per chunk, reducing the chance of missing key information. |
| **Context Window Usage** | ✅ Efficient — you can fit more distinct chunks in the LLM context window (e.g., 10 × 256 = 2,560 tokens vs 3 × 1024 = 3,072). | ❌ Less efficient — fewer chunks fit, and much of the content may be irrelevant filler. |
| **Embedding Quality** | ⚠️ May be too short for sentence-transformer models that work best with paragraph-length text. | ⚠️ May average out meaning, making the embedding less precise. |
| **Generation Quality** | ❌ LLM may lack enough context to produce coherent answers. | ✅ LLM has richer context for each retrieved chunk. |

**Best practice**: Use 512 tokens as a starting point with 50–100 token overlap, then tune based on evaluation metrics. Different document types may need different strategies — code documentation benefits from smaller chunks, while narrative text works better with larger ones.

---

### Question 3 (Conceptual)
**What is the difference between naive RAG and hybrid search? Why might hybrid outperform pure semantic search?**

**Naive RAG** uses a single retrieval method — typically pure semantic (vector) search:
1. Query → Embed with the same model used for documents
2. Find top-k nearest neighbours in the vector database (cosine similarity)
3. Pass retrieved chunks + query to the LLM for answer generation

**Hybrid search** combines **two or more retrieval strategies** — most commonly:
- **BM25 (keyword search)**: A statistical method using term frequency and inverse document frequency. Excels at exact term matching.
- **Semantic search (vector)**: Uses embedding similarity. Excels at understanding meaning and synonyms.

These are combined using techniques like **Reciprocal Rank Fusion (RRF)** or **weighted scoring**.

**Why hybrid outperforms pure semantic search:**

1. **Exact-match gaps**: Semantic search can miss queries where exact keywords matter (e.g., searching for error code "ERR_CONNECTION_REFUSED" — semantic models may not understand this as a precise string).
2. **Rare terms**: Embedding models may not handle rare or domain-specific terms well, but BM25 will find exact matches.
3. **Complementary strengths**: Semantic search finds conceptually similar text ("the dog ran" ↔ "the canine sprinted"), while BM25 catches literal matches. Together, they cover more retrieval scenarios.
4. **Re-ranking opportunity**: After hybrid retrieval produces a diverse candidate set, a cross-encoder re-ranker can make the final ranking more accurate than either method alone.

---

### Question 4 (Technical)
**Name three RAGAS metrics and explain what each one measures about a RAG system.**

1. **Faithfulness**: Measures whether the generated answer is factually consistent with the retrieved context. It checks if every claim in the answer can be traced back to the provided context chunks. A high faithfulness score means the LLM is not hallucinating — it's only stating things that are supported by the retrieved documents. Formula: `faithfulness = (number of claims supported by context) / (total number of claims in the answer)`.

2. **Answer Relevancy**: Measures how relevant and complete the generated answer is to the original question. It uses an LLM to generate synthetic questions from the answer, then computes cosine similarity between those synthetic questions and the original question. If the answer is highly relevant, the generated questions should be semantically close to the original. A low score indicates the answer is off-topic, includes unnecessary information, or is incomplete.

3. **Context Precision**: Evaluates whether the retrieved context chunks that are actually relevant to the question are ranked higher than irrelevant ones. It's essentially a precision metric over the retrieval step. High context precision means the retrieval system is putting the most useful chunks at the top of the results — which matters because LLMs tend to attend more to information at the beginning of the context window. Formula: `context_precision = weighted average of (relevant chunks in top-k positions)`.

---

### Question 5 (Technical)
**What does a cross-encoder re-ranker do, and why is it applied after initial retrieval rather than instead of it?**

A **cross-encoder re-ranker** is a transformer model that takes a **(query, document) pair** as input and outputs a relevance score. Unlike bi-encoders (used in semantic search) that encode the query and document independently, a cross-encoder processes them **jointly** — allowing deep attention between query and document tokens. This makes cross-encoders significantly more accurate at judging relevance.

**Why it's applied AFTER initial retrieval rather than INSTEAD of it:**

1. **Computational cost**: Cross-encoders must process every (query, document) pair individually. If you have 100,000 documents, you'd need 100,000 forward passes per query — this is prohibitively slow. A bi-encoder can pre-compute all document embeddings once and use fast approximate nearest-neighbor search.

2. **Two-stage architecture**:
   - **Stage 1 (Retrieval)**: Use a fast bi-encoder or BM25 to retrieve a small candidate set (e.g., top 20–50 documents) from millions.
   - **Stage 2 (Re-ranking)**: Use the cross-encoder to precisely re-rank only those 20–50 candidates. This is tractable because the set is small.

3. **Accuracy vs. speed tradeoff**: Bi-encoders are fast but approximate. Cross-encoders are slow but precise. The two-stage approach gives you the best of both worlds: broad recall from fast retrieval + precise ranking from the cross-encoder.

4. **Practical impact**: In experiments, cross-encoder re-ranking typically improves top-1 accuracy by 5–15% over bi-encoder retrieval alone, with minimal latency increase (50–200ms for re-ranking 20 documents).

---

### Question 6 (Design)
**Design a RAG architecture for a multi-tenant SaaS product where each customer's documents must remain isolated.**

```
┌─────────────────────────────────────────────────────────────────────┐
│                    Multi-Tenant RAG Architecture                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────────┐     ┌──────────────────────────────────────────┐  │
│  │  API Gateway  │────▶│  Authentication & Tenant Resolution      │  │
│  │  (FastAPI)    │     │  • JWT token validation                  │  │
│  └──────────────┘     │  • Extract tenant_id from token          │  │
│                        │  • Rate limiting per tenant               │  │
│                        └──────────────────────────────────────────┘  │
│                                      │                               │
│                    ┌─────────────────┴─────────────────┐            │
│                    ▼                                   ▼            │
│  ┌─────────────────────────┐   ┌─────────────────────────────┐     │
│  │  Document Ingestion     │   │  Query Pipeline              │     │
│  │  ───────────────────    │   │  ───────────────────         │     │
│  │  1. Upload (S3/local)   │   │  1. Embed query              │     │
│  │  2. Parse (PDF/DOCX)    │   │  2. Retrieve from TENANT     │     │
│  │  3. Chunk (512 tokens)  │   │     collection ONLY          │     │
│  │  4. Embed (sent-trans)  │   │  3. Hybrid search + rerank   │     │
│  │  5. Store in TENANT     │   │  4. Generate answer (LLM)    │     │
│  │     namespace           │   │  5. Return with citations    │     │
│  └─────────────────────────┘   └─────────────────────────────┘     │
│                                                                      │
│  ┌──────────────────────────────────────────────────────────────┐   │
│  │  Vector Database (ChromaDB / Qdrant)                          │   │
│  │  ─────────────────────────────────────                        │   │
│  │  Collection: tenant_001_docs  ◄── Isolated namespace          │   │
│  │  Collection: tenant_002_docs  ◄── per customer                │   │
│  │  Collection: tenant_003_docs  ◄── No cross-tenant access      │   │
│  └──────────────────────────────────────────────────────────────┘   │
│                                                                      │
│  Key Isolation Strategies:                                           │
│  • Separate ChromaDB collections per tenant (namespace isolation)   │
│  • All queries scoped to tenant_id — enforced at service layer      │
│  • Encryption at rest per tenant (AES-256)                          │
│  • Row-level security if using PostgreSQL for metadata              │
│  • Audit logging: every query and document access is logged         │
│  • Data retention policies configurable per tenant                  │
└─────────────────────────────────────────────────────────────────────┘
```

**Key design decisions:**

1. **Namespace isolation (not filtering)**: Using separate ChromaDB collections per tenant rather than metadata filtering within a shared collection. This prevents accidental data leakage and is simpler to reason about.

2. **Authentication layer**: Every request must include a JWT token that resolves to a `tenant_id`. The service layer enforces that queries only access the tenant's own collection.

3. **Background processing**: Document ingestion runs asynchronously via a task queue (Celery/asyncio). The user uploads a document, receives a job ID, and polls for status.

4. **Scalability**: Each tenant's collection can be independently scaled. Large tenants can get dedicated vector database instances.

5. **Audit trail**: Every query and retrieval operation is logged with `tenant_id`, `timestamp`, `query`, and `documents_accessed` for compliance.
