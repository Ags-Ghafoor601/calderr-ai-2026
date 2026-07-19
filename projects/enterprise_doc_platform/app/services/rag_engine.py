"""Enterprise Document Intelligence Platform — Advanced RAG Engine.

Implements hybrid search (BM25 + semantic) with cross-encoder re-ranking
and confidence scoring for production-quality RAG responses.
"""

import logging
import time
from typing import Optional

import numpy as np
from groq import Groq

from app.config import settings
from app.services.vector_store import vector_store

logger = logging.getLogger(__name__)


class RAGEngine:
    """Advanced RAG engine with hybrid search and re-ranking."""

    def __init__(self):
        self._groq = Groq(api_key=settings.groq_api_key)
        self._reranker = None
        self._bm25_cache: dict[str, object] = {}

    # ---- BM25 Retrieval ----
    def _get_bm25(self, tenant_id: str):
        """Get or build BM25 index for a tenant."""
        from rank_bm25 import BM25Okapi

        if tenant_id not in self._bm25_cache:
            chunks = vector_store.get_all_chunks_for_tenant(tenant_id)
            if not chunks:
                return None, []

            corpus = [c["text"].lower().split() for c in chunks]
            bm25 = BM25Okapi(corpus)
            self._bm25_cache[tenant_id] = (bm25, chunks)
            logger.info("Built BM25 index for tenant '%s' (%d docs)", tenant_id, len(chunks))

        return self._bm25_cache[tenant_id]

    def invalidate_bm25(self, tenant_id: str):
        """Clear cached BM25 index when documents change."""
        self._bm25_cache.pop(tenant_id, None)

    def _bm25_retrieve(self, tenant_id: str, query: str, top_k: int = 10) -> list[dict]:
        """Retrieve documents using BM25 keyword search."""
        result = self._get_bm25(tenant_id)
        if result is None:
            return []

        bm25, chunks = result
        tokenized_query = query.lower().split()
        scores = bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append({
                    "text": chunks[idx]["text"],
                    "metadata": chunks[idx]["metadata"],
                    "score": float(scores[idx]),
                    "method": "bm25",
                })
        return results

    # ---- Cross-Encoder Re-ranking ----
    def _rerank(self, query: str, results: list[dict], top_k: int = 5) -> list[dict]:
        """Re-rank results using a cross-encoder model."""
        if not results:
            return results

        from sentence_transformers import CrossEncoder

        if self._reranker is None:
            logger.info("Loading cross-encoder re-ranker...")
            self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        pairs = [(query, r["text"]) for r in results]
        scores = self._reranker.predict(pairs)

        for i, score in enumerate(scores):
            results[i]["rerank_score"] = float(score)
            results[i]["method"] = results[i].get("method", "") + "+rerank"

        reranked = sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)
        return reranked[:top_k]

    # ---- Hybrid Retrieval ----
    def hybrid_retrieve(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        use_reranking: bool = True,
    ) -> list[dict]:
        """Perform hybrid retrieval with optional re-ranking.

        Args:
            tenant_id: Tenant to search
            query: User's question
            top_k: Number of final results
            use_hybrid: Whether to combine BM25 + semantic
            use_reranking: Whether to apply cross-encoder re-ranking

        Returns:
            List of retrieved chunks with scores.
        """
        # Semantic retrieval (always)
        semantic_results = vector_store.semantic_search(
            tenant_id, query, top_k=top_k * 2
        )

        if not use_hybrid:
            if use_reranking and semantic_results:
                return self._rerank(query, semantic_results, top_k)
            return semantic_results[:top_k]

        # BM25 retrieval
        bm25_results = self._bm25_retrieve(tenant_id, query, top_k=top_k * 2)

        # Reciprocal Rank Fusion
        doc_scores: dict[str, dict] = {}

        for rank, r in enumerate(semantic_results):
            key = r["text"][:100]
            rrf_score = settings.semantic_weight / (rank + 1)
            if key in doc_scores:
                doc_scores[key]["score"] += rrf_score
            else:
                doc_scores[key] = {**r, "score": rrf_score, "method": "hybrid"}

        for rank, r in enumerate(bm25_results):
            key = r["text"][:100]
            rrf_score = settings.bm25_weight / (rank + 1)
            if key in doc_scores:
                doc_scores[key]["score"] += rrf_score
            else:
                doc_scores[key] = {**r, "score": rrf_score, "method": "hybrid"}

        merged = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)

        if use_reranking and merged:
            return self._rerank(query, merged[:top_k * 2], top_k)

        return merged[:top_k]

    # ---- Answer Generation ----
    def generate_answer(self, query: str, chunks: list[dict]) -> tuple[str, float]:
        """Generate an answer using Groq LLM with confidence scoring.

        Returns:
            Tuple of (answer_text, confidence_score).
        """
        if not chunks:
            return "No relevant documents found to answer this question.", 0.0

        context = "\n\n---\n\n".join(
            f"[Source: {c['metadata'].get('source', '?')}, "
            f"Page {c['metadata'].get('page', '?')}]\n{c['text']}"
            for c in chunks
        )

        prompt = f"""You are an enterprise document intelligence assistant.
Answer the question based ONLY on the provided context.
If the context doesn't contain sufficient information, clearly state that.
Always cite the source document and page number.
Be precise, professional, and concise.

Context:
{context}

Question: {query}

Provide your answer followed by a confidence assessment.
Format:
ANSWER: [your detailed answer with citations]
CONFIDENCE: [HIGH/MEDIUM/LOW] - [brief reason]"""

        response = self._groq.chat.completions.create(
            model=settings.llm_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )

        full_response = response.choices[0].message.content

        # Parse confidence
        answer = full_response
        confidence = 0.7  # Default

        if "CONFIDENCE:" in full_response:
            parts = full_response.split("CONFIDENCE:")
            answer = parts[0].replace("ANSWER:", "").strip()
            conf_text = parts[1].strip().upper()
            if "HIGH" in conf_text:
                confidence = 0.9
            elif "MEDIUM" in conf_text:
                confidence = 0.7
            elif "LOW" in conf_text:
                confidence = 0.4
        elif "ANSWER:" in full_response:
            answer = full_response.replace("ANSWER:", "").strip()

        # Adjust confidence based on retrieval scores
        if chunks:
            avg_score = np.mean([c.get("score", 0.5) for c in chunks])
            confidence = min(confidence, max(avg_score, 0.1))

        return answer, round(confidence, 3)

    # ---- Full RAG Query ----
    def query(
        self,
        tenant_id: str,
        question: str,
        top_k: int = 5,
        use_hybrid: bool = True,
        use_reranking: bool = True,
    ) -> dict:
        """Execute a full RAG query and return structured response.

        Returns dict with: answer, confidence, chunks, method, time_ms
        """
        start = time.perf_counter()

        # Retrieve
        chunks = self.hybrid_retrieve(
            tenant_id, question,
            top_k=top_k,
            use_hybrid=use_hybrid,
            use_reranking=use_reranking,
        )

        method = "semantic"
        if use_hybrid:
            method = "hybrid"
        if use_reranking:
            method += "+rerank"

        # Generate
        answer, confidence = self.generate_answer(question, chunks)

        elapsed_ms = (time.perf_counter() - start) * 1000

        return {
            "question": question,
            "answer": answer,
            "confidence_score": confidence,
            "source_chunks": [
                {
                    "text": c["text"],
                    "source_file": c["metadata"].get("source", "unknown"),
                    "page": c["metadata"].get("page", 0),
                    "similarity_score": round(c.get("score", 0), 4),
                    "chunk_index": c["metadata"].get("chunk_index", 0),
                }
                for c in chunks
            ],
            "retrieval_method": method,
            "processing_time_ms": round(elapsed_ms, 1),
            "tenant_id": tenant_id,
        }


# Module-level singleton
rag_engine = RAGEngine()
