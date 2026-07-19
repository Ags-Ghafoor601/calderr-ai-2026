#!/usr/bin/env python3
"""
Lab 3.3 — Hybrid Retrieval & RAGAS Evaluation
===============================================
Implements hybrid search combining BM25 keyword search with semantic retrieval.
Adds cross-encoder re-ranking and multi-query retrieval.
Evaluates the full pipeline using RAGAS metrics.

Features:
  • BM25 keyword search (rank_bm25)
  • Semantic search (ChromaDB + sentence-transformers)
  • Ensemble retrieval combining both approaches
  • Cross-encoder re-ranking (cross-encoder/ms-marco-MiniLM-L-6-v2)
  • Multi-query generation (3 query variations)
  • RAGAS evaluation: faithfulness, answer relevancy, context precision
  • Comparative analysis: naive RAG vs hybrid vs hybrid+rerank

Usage:
    python lab_3_3_hybrid_rag.py ingest             # Ingest documents
    python lab_3_3_hybrid_rag.py query "question"    # Hybrid RAG query
    python lab_3_3_hybrid_rag.py compare "question"  # Compare naive vs hybrid
    python lab_3_3_hybrid_rag.py evaluate            # Full RAGAS evaluation
"""

import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# Fix Windows console encoding for Rich Unicode output (spinners, braille chars)
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich import box

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

console = Console()
app = typer.Typer(
    name="hybrid-rag",
    help="🔀 Hybrid RAG Pipeline — Lab 3.3",
    add_completion=False,
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_DIR = ROOT_DIR / "labs" / ".chromadb_lab33"

# ---------------------------------------------------------------------------
# Evaluation Q&A set with ground truth contexts
# ---------------------------------------------------------------------------
EVAL_DATASET = [
    {
        "question": "What is the primary theme of Week 3?",
        "ground_truth": "Week 3 focuses on Embeddings, RAG and Vector Databases.",
        "ground_truth_context": "Week 3 introduces the technology that lets AI systems work with knowledge they were never trained on, covering embeddings, vector databases, and retrieval-augmented generation.",
    },
    {
        "question": "What embedding models are compared in Lab 3.1?",
        "ground_truth": "all-MiniLM-L6-v2 and bge-small-en are compared.",
        "ground_truth_context": "Lab 3.1 compares results between all-MiniLM-L6-v2 and bge-small-en embedding models for semantic search.",
    },
    {
        "question": "What chunk sizes are tested in Lab 3.2?",
        "ground_truth": "Chunk sizes of 256, 512, and 1024 tokens are tested.",
        "ground_truth_context": "Tune chunk size across 256, 512, and 1024 tokens and document the impact on answer quality.",
    },
    {
        "question": "What is the purpose of a cross-encoder re-ranker?",
        "ground_truth": "A cross-encoder re-ranker scores query-document pairs jointly to improve ranking of retrieved results.",
        "ground_truth_context": "Add cross-encoder reranking using BAAI/bge-reranker-base to re-rank retrieved documents for better relevance.",
    },
    {
        "question": "What are the main RAGAS metrics used for evaluation?",
        "ground_truth": "RAGAS metrics include faithfulness, answer relevancy, context precision, and context recall.",
        "ground_truth_context": "Run faithfulness, answer relevancy, context precision metrics on your RAG system using the RAGAS framework.",
    },
    {
        "question": "What vector databases are covered in Week 3?",
        "ground_truth": "ChromaDB, FAISS, and Qdrant are the vector databases covered.",
        "ground_truth_context": "Work with ChromaDB, FAISS, and Qdrant vector databases for storing and retrieving embeddings.",
    },
    {
        "question": "What is the difference between naive RAG and advanced RAG?",
        "ground_truth": "Naive RAG uses simple retrieval while advanced RAG adds hybrid search, re-ranking, and multi-query techniques.",
        "ground_truth_context": "Study hybrid search combining BM25 keyword search with semantic search, implement with EnsembleRetriever, and add cross-encoder reranking.",
    },
    {
        "question": "How many documents should the Personal Knowledge Base ingest?",
        "ground_truth": "The Personal Knowledge Base should ingest 20 or more documents.",
        "ground_truth_context": "CLI tool ingesting 20+ documents, 15 Q&A examples, README with architecture.",
    },
    {
        "question": "What is the total weekly commitment for the internship?",
        "ground_truth": "The total commitment is 20 hours per week.",
        "ground_truth_context": "Total Commitment: 20 hours (4 hours/day for 5 days).",
    },
    {
        "question": "What should be included in the Friday standup demo?",
        "ground_truth": "The demo should include live RAG queries, technical questions, architecture review, RAGAS scores, and blockers.",
        "ground_truth_context": "Live demo of your RAG system (2 minutes). Run at least 3 real queries against your indexed documents.",
    },
]


# ---------------------------------------------------------------------------
# PDF Loading (reuses logic from Lab 3.2)
# ---------------------------------------------------------------------------
def load_pdfs(pdf_dir: Path) -> list[dict]:
    """Load CalderR PDFs and return document chunks."""
    import fitz

    documents = []
    pdf_files = sorted(pdf_dir.glob("CalderR_*.pdf"))
    if not pdf_files:
        console.print("[red]No CalderR PDFs found![/red]")
        raise typer.Exit(1)

    for pdf_path in pdf_files:
        try:
            doc = fitz.open(str(pdf_path))
            for page_num, page in enumerate(doc, 1):
                text = page.get_text().strip()
                if text and len(text) > 50:
                    documents.append({
                        "text": text,
                        "metadata": {
                            "source": pdf_path.name,
                            "page": page_num,
                        },
                    })
            doc.close()
        except Exception as e:
            console.print(f"  [yellow]⚠ Skipped {pdf_path.name}: {e}[/yellow]")

    console.print(f"  [green]✓[/green] Loaded {len(documents)} pages from {len(pdf_files)} PDFs")
    return documents


def split_documents(documents: list[dict], chunk_size: int = 512, chunk_overlap: int = 50) -> list[dict]:
    """Split documents into chunks."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["text"])
        for i, text in enumerate(splits):
            chunks.append({
                "text": text,
                "metadata": {**doc["metadata"], "chunk_index": i},
            })

    console.print(f"  [green]✓[/green] Split into {len(chunks)} chunks")
    return chunks


# ---------------------------------------------------------------------------
# BM25 Index
# ---------------------------------------------------------------------------
class BM25Retriever:
    """BM25 keyword-based retriever."""

    def __init__(self, chunks: list[dict]):
        from rank_bm25 import BM25Okapi

        self.chunks = chunks
        self.corpus = [c["text"].lower().split() for c in chunks]
        self.bm25 = BM25Okapi(self.corpus)
        console.print(f"  [green]✓[/green] BM25 index built with {len(chunks)} documents")

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "text": self.chunks[idx]["text"],
                "metadata": self.chunks[idx]["metadata"],
                "score": float(scores[idx]),
                "method": "bm25",
            })
        return results


# ---------------------------------------------------------------------------
# Semantic Retriever (ChromaDB)
# ---------------------------------------------------------------------------
class SemanticRetriever:
    """ChromaDB-based semantic retriever."""

    def __init__(self, chunks: list[dict], collection_name: str = "hybrid_rag"):
        import chromadb
        from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

        self.ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        self.client = chromadb.PersistentClient(path=str(CHROMA_DIR))

        try:
            self.client.delete_collection(collection_name)
        except Exception:
            pass

        self.collection = self.client.create_collection(
            name=collection_name,
            embedding_function=self.ef,
            metadata={"hnsw:space": "cosine"},
        )

        batch_size = 500
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i + batch_size]
            self.collection.add(
                ids=[f"chunk_{i + j}" for j in range(len(batch))],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
            )

        console.print(
            f"  [green]✓[/green] Semantic index built with {self.collection.count()} documents"
        )

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        results = self.collection.query(query_texts=[query], n_results=top_k)
        retrieved = []
        for i in range(len(results["documents"][0])):
            retrieved.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "score": 1 - results["distances"][0][i] if results.get("distances") else 0.5,
                "method": "semantic",
            })
        return retrieved


# ---------------------------------------------------------------------------
# Hybrid Retriever (Ensemble)
# ---------------------------------------------------------------------------
class HybridRetriever:
    """Combines BM25 and semantic retrieval with optional re-ranking."""

    def __init__(self, bm25: BM25Retriever, semantic: SemanticRetriever):
        self.bm25 = bm25
        self.semantic = semantic
        self._reranker = None

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        bm25_weight: float = 0.4,
        semantic_weight: float = 0.6,
        rerank: bool = False,
    ) -> list[dict]:
        """Retrieve using ensemble of BM25 + semantic, optionally re-rank."""
        # Get candidates from both retrievers
        bm25_results = self.bm25.retrieve(query, top_k=top_k * 2)
        semantic_results = self.semantic.retrieve(query, top_k=top_k * 2)

        # Merge by reciprocal rank fusion
        doc_scores: dict[str, dict] = {}

        for rank, r in enumerate(bm25_results):
            key = r["text"][:100]
            rrf_score = bm25_weight / (rank + 1)
            if key in doc_scores:
                doc_scores[key]["score"] += rrf_score
            else:
                doc_scores[key] = {**r, "score": rrf_score, "method": "hybrid"}

        for rank, r in enumerate(semantic_results):
            key = r["text"][:100]
            rrf_score = semantic_weight / (rank + 1)
            if key in doc_scores:
                doc_scores[key]["score"] += rrf_score
            else:
                doc_scores[key] = {**r, "score": rrf_score, "method": "hybrid"}

        # Sort by combined score
        merged = sorted(doc_scores.values(), key=lambda x: x["score"], reverse=True)
        merged = merged[:top_k * 2]  # Keep more for potential re-ranking

        if rerank and merged:
            merged = self._cross_encoder_rerank(query, merged)

        return merged[:top_k]

    def _cross_encoder_rerank(self, query: str, results: list[dict]) -> list[dict]:
        """Re-rank results using a cross-encoder model."""
        from sentence_transformers import CrossEncoder

        if self._reranker is None:
            with Progress(
                SpinnerColumn(),
                TextColumn("[bold yellow]Loading cross-encoder re-ranker..."),
                console=console,
            ) as progress:
                progress.add_task("load")
                self._reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        pairs = [(query, r["text"]) for r in results]
        scores = self._reranker.predict(pairs)

        for i, score in enumerate(scores):
            results[i]["rerank_score"] = float(score)
            results[i]["method"] = "hybrid+rerank"

        return sorted(results, key=lambda x: x.get("rerank_score", 0), reverse=True)


# ---------------------------------------------------------------------------
# Multi-Query Generator
# ---------------------------------------------------------------------------
def generate_query_variations(query: str, n: int = 3) -> list[str]:
    """Generate n query variations using Groq LLM."""
    from groq import Groq

    client = Groq(api_key=GROQ_API_KEY)
    prompt = f"""Generate {n} alternative versions of the following search query.
Each version should approach the same information need from a different angle.
Return ONLY the queries, one per line, numbered 1-{n}.

Original query: {query}"""

    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=256,
    )

    variations = [query]  # Always include original
    for line in response.choices[0].message.content.strip().split("\n"):
        cleaned = line.strip().lstrip("0123456789.-) ").strip()
        if cleaned and len(cleaned) > 10:
            variations.append(cleaned)

    return variations[:n + 1]


def multi_query_retrieve(
    retriever: HybridRetriever, query: str, top_k: int = 5, rerank: bool = True
) -> list[dict]:
    """Retrieve using multiple query variations, deduplicate, and re-rank."""
    variations = generate_query_variations(query, n=3)

    console.print(f"  [dim]Generated {len(variations)} query variations:[/dim]")
    for i, v in enumerate(variations):
        console.print(f"    [dim]{i+1}. {v}[/dim]")

    all_results: dict[str, dict] = {}
    for v in variations:
        results = retriever.retrieve(v, top_k=top_k, rerank=rerank)
        for r in results:
            key = r["text"][:100]
            if key not in all_results or r["score"] > all_results[key]["score"]:
                all_results[key] = r

    merged = sorted(all_results.values(), key=lambda x: x["score"], reverse=True)
    return merged[:top_k]


# ---------------------------------------------------------------------------
# Answer Generation
# ---------------------------------------------------------------------------
def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """Generate answer using Groq LLM."""
    from groq import Groq

    context = "\n\n---\n\n".join(
        f"[Source: {c['metadata'].get('source', '?')}, Page {c['metadata'].get('page', '?')}]\n{c['text']}"
        for c in context_chunks
    )

    prompt = f"""You are a helpful assistant. Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say so clearly.
Always cite the source document and page number.

Context:
{context}

Question: {query}

Answer:"""

    client = Groq(api_key=GROQ_API_KEY)
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=1024,
    )
    return response.choices[0].message.content


# ---------------------------------------------------------------------------
# RAGAS Evaluation
# ---------------------------------------------------------------------------
def run_ragas_evaluation(
    retriever,
    eval_data: list[dict],
    mode: str = "hybrid",
    rerank: bool = False,
) -> dict:
    """Run RAGAS evaluation and return metrics."""
    console.print(f"\n  [bold cyan]Running RAGAS evaluation (mode={mode}, rerank={rerank})...[/bold cyan]")

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    with Progress(
        SpinnerColumn(),
        TextColumn(f"[cyan]Generating answers ({mode})..."),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("eval", total=len(eval_data))

        for item in eval_data:
            try:
                if hasattr(retriever, "retrieve"):
                    if isinstance(retriever, HybridRetriever):
                        chunks = retriever.retrieve(item["question"], top_k=5, rerank=rerank)
                    else:
                        chunks = retriever.retrieve(item["question"], top_k=5)
                else:
                    chunks = retriever.retrieve(item["question"], top_k=5)

                answer = generate_answer(item["question"], chunks)
                ctx = [c["text"] for c in chunks]

                questions.append(item["question"])
                answers.append(answer)
                contexts.append(ctx)
                ground_truths.append(item["ground_truth"])

                time.sleep(0.5)  # Rate limit
            except Exception as e:
                console.print(f"  [yellow]⚠ Error on: {item['question'][:50]}... → {e}[/yellow]")
                questions.append(item["question"])
                answers.append("Error generating answer.")
                contexts.append(["No context retrieved."])
                ground_truths.append(item["ground_truth"])

            progress.advance(task)

    # Run RAGAS evaluation
    try:
        from datasets import Dataset

        eval_dataset = Dataset.from_dict({
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        })

        # Try RAGAS evaluation with Groq LLM
        try:
            from ragas import evaluate as ragas_evaluate
            from ragas.metrics import (
                faithfulness,
                answer_relevancy,
                context_precision,
            )
            from langchain_groq import ChatGroq
            from langchain_huggingface import HuggingFaceEmbeddings

            llm = ChatGroq(
                model="llama-3.3-70b-versatile",
                api_key=GROQ_API_KEY,
                temperature=0,
            )
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

            result = ragas_evaluate(
                dataset=eval_dataset,
                metrics=[faithfulness, answer_relevancy, context_precision],
                llm=llm,
                embeddings=embeddings,
            )

            metrics = {
                "faithfulness": float(result.get("faithfulness", 0)),
                "answer_relevancy": float(result.get("answer_relevancy", 0)),
                "context_precision": float(result.get("context_precision", 0)),
                "mode": mode,
                "rerank": rerank,
            }

            console.print(f"  [green]✓ RAGAS evaluation complete[/green]")
            return metrics

        except Exception as e:
            console.print(f"  [yellow]⚠ RAGAS library evaluation failed: {e}[/yellow]")
            console.print("  [dim]Falling back to custom evaluation metrics...[/dim]")

            # Custom evaluation fallback
            return _custom_evaluation(questions, answers, contexts, ground_truths, mode, rerank)

    except Exception as e:
        console.print(f"  [red]Evaluation error: {e}[/red]")
        return _custom_evaluation(questions, answers, contexts, ground_truths, mode, rerank)


def _custom_evaluation(
    questions: list, answers: list, contexts: list,
    ground_truths: list, mode: str, rerank: bool,
) -> dict:
    """Custom fallback evaluation metrics mimicking RAGAS."""
    faithfulness_scores = []
    relevancy_scores = []
    context_precision_scores = []

    for i in range(len(questions)):
        # Faithfulness: How much of the answer is grounded in context?
        answer_terms = set(answers[i].lower().split())
        context_text = " ".join(contexts[i]).lower()
        context_terms = set(context_text.split())
        grounded = len(answer_terms & context_terms) / max(len(answer_terms), 1)
        faithfulness_scores.append(min(grounded * 1.5, 1.0))

        # Answer Relevancy: How relevant is the answer to the question?
        question_terms = set(questions[i].lower().split())
        ans_relevant = len(question_terms & answer_terms) / max(len(question_terms), 1)
        gt_overlap = len(set(ground_truths[i].lower().split()) & answer_terms) / max(
            len(set(ground_truths[i].lower().split())), 1
        )
        relevancy_scores.append(min((ans_relevant + gt_overlap) / 2 * 2.0, 1.0))

        # Context Precision: Is the context relevant to the question?
        ctx_relevant = len(question_terms & context_terms) / max(len(question_terms), 1)
        context_precision_scores.append(min(ctx_relevant * 1.5, 1.0))

    return {
        "faithfulness": float(np.mean(faithfulness_scores)),
        "answer_relevancy": float(np.mean(relevancy_scores)),
        "context_precision": float(np.mean(context_precision_scores)),
        "mode": mode,
        "rerank": rerank,
        "evaluation_method": "custom",
    }


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def display_results(results: list[dict], query: str, method: str):
    """Display retrieval results."""
    console.print()
    console.print(Panel(
        f"[bold cyan]Query:[/bold cyan] {query}\n[dim]Method: {method}[/dim]",
        border_style="bright_blue",
    ))

    table = Table(
        title=f"🔍 {method.upper()} Results",
        box=box.ROUNDED,
        border_style="bright_blue",
        show_lines=True,
    )
    table.add_column("#", style="bold cyan", width=3)
    table.add_column("Score", style="green", width=10)
    table.add_column("Method", style="yellow", width=15)
    table.add_column("Source", style="magenta", width=20)
    table.add_column("Text Preview", style="white", ratio=1)

    for i, r in enumerate(results, 1):
        score = r.get("rerank_score", r.get("score", 0))
        table.add_row(
            str(i),
            f"{score:.4f}",
            r.get("method", "?"),
            f"{r['metadata'].get('source', '?')}:p{r['metadata'].get('page', '?')}",
            r["text"][:80].replace("\n", " ") + "...",
        )

    console.print(table)


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------
@app.command()
def ingest():
    """Ingest CalderR PDFs and build BM25 + semantic indices."""
    console.print(Panel(
        "[bold]🔀 Hybrid RAG — Document Ingestion[/bold]",
        style="bright_blue",
    ))

    documents = load_pdfs(ROOT_DIR)
    chunks = split_documents(documents, chunk_size=512, chunk_overlap=50)

    # Save chunks for BM25 (needs to be rebuilt each time from data)
    chunks_path = ROOT_DIR / "labs" / ".chunks_lab33.json"
    with open(chunks_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)

    # Build semantic index
    SemanticRetriever(chunks)
    console.print("\n[bold green]✅ Ingestion complete![/bold green]")
    console.print(f"  [dim]Chunks saved to {chunks_path}[/dim]")


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    rerank: bool = typer.Option(True, "--rerank/--no-rerank"),
    multi_query: bool = typer.Option(False, "--multi-query", "-mq"),
):
    """Ask a question using hybrid retrieval."""
    chunks = _load_chunks()
    bm25 = BM25Retriever(chunks)
    semantic = SemanticRetriever(chunks, collection_name="hybrid_query")
    hybrid = HybridRetriever(bm25, semantic)

    if multi_query:
        results = multi_query_retrieve(hybrid, question, top_k=top_k, rerank=rerank)
    else:
        results = hybrid.retrieve(question, top_k=top_k, rerank=rerank)

    method = "hybrid" + ("+rerank" if rerank else "") + ("+multiquery" if multi_query else "")
    display_results(results, question, method)

    answer = generate_answer(question, results)
    console.print(Panel(
        f"[bold green]Answer:[/bold green]\n{answer}",
        border_style="green",
        title="💡 Hybrid RAG Response",
    ))


@app.command()
def compare(
    question: str = typer.Argument(..., help="Question to compare"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
):
    """Compare naive semantic, BM25, hybrid, and hybrid+rerank retrieval."""
    chunks = _load_chunks()
    bm25 = BM25Retriever(chunks)
    semantic = SemanticRetriever(chunks, collection_name="compare_semantic")
    hybrid = HybridRetriever(bm25, semantic)

    console.print(Panel(
        f"[bold]🔬 Retrieval Method Comparison[/bold]\n"
        f"Query: [italic]\"{question}\"[/italic]",
        style="bright_magenta",
    ))

    # 1. BM25 only
    bm25_results = bm25.retrieve(question, top_k=top_k)
    display_results(bm25_results, question, "BM25 (Keyword)")

    # 2. Semantic only
    sem_results = semantic.retrieve(question, top_k=top_k)
    display_results(sem_results, question, "Semantic (Vector)")

    # 3. Hybrid (no rerank)
    hybrid_results = hybrid.retrieve(question, top_k=top_k, rerank=False)
    display_results(hybrid_results, question, "Hybrid (Ensemble)")

    # 4. Hybrid + rerank
    hybrid_rerank = hybrid.retrieve(question, top_k=top_k, rerank=True)
    display_results(hybrid_rerank, question, "Hybrid + Re-rank")


@app.command()
def evaluate():
    """Run full RAGAS evaluation comparing naive vs hybrid vs hybrid+rerank."""
    console.print(Panel(
        "[bold]📊 RAGAS Evaluation Suite[/bold]\n"
        "Comparing: Semantic only → Hybrid → Hybrid + Re-rank\n"
        f"Evaluation set: {len(EVAL_DATASET)} Q&A pairs",
        style="bright_magenta",
    ))

    chunks = _load_chunks()
    bm25 = BM25Retriever(chunks)
    semantic = SemanticRetriever(chunks, collection_name="eval_semantic")
    hybrid = HybridRetriever(bm25, semantic)

    all_metrics = {}

    # 1. Semantic only
    console.print(f"\n{'='*60}")
    console.print("[bold]1/3 — Semantic Only[/bold]")
    all_metrics["semantic"] = run_ragas_evaluation(
        semantic, EVAL_DATASET, mode="semantic", rerank=False
    )

    # 2. Hybrid (no rerank)
    console.print(f"\n{'='*60}")
    console.print("[bold]2/3 — Hybrid (BM25 + Semantic)[/bold]")
    all_metrics["hybrid"] = run_ragas_evaluation(
        hybrid, EVAL_DATASET, mode="hybrid", rerank=False
    )

    # 3. Hybrid + rerank
    console.print(f"\n{'='*60}")
    console.print("[bold]3/3 — Hybrid + Cross-Encoder Re-rank[/bold]")
    all_metrics["hybrid_rerank"] = run_ragas_evaluation(
        hybrid, EVAL_DATASET, mode="hybrid+rerank", rerank=True
    )

    # Summary comparison table
    console.print(f"\n{'='*60}")
    table = Table(
        title="📊 RAGAS Evaluation Comparison",
        box=box.DOUBLE_EDGE,
        title_style="bold magenta",
        border_style="bright_blue",
        show_lines=True,
    )
    table.add_column("Metric", style="bold white")
    table.add_column("Semantic Only", style="cyan", justify="center")
    table.add_column("Hybrid", style="yellow", justify="center")
    table.add_column("Hybrid+Rerank", style="green", justify="center")

    for metric_name in ["faithfulness", "answer_relevancy", "context_precision"]:
        row = [metric_name.replace("_", " ").title()]
        for mode in ["semantic", "hybrid", "hybrid_rerank"]:
            val = all_metrics.get(mode, {}).get(metric_name, 0)
            color = "green" if val > 0.7 else "yellow" if val > 0.4 else "red"
            row.append(f"[{color}]{val:.4f}[/{color}]")
        table.add_row(*row)

    console.print(table)

    # Save results
    report_path = ROOT_DIR / "labs" / "lab_3_3_ragas_report.json"
    with open(report_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    console.print(f"\n  [dim]Report saved → {report_path}[/dim]")
    console.print("[bold green]✅ RAGAS evaluation complete![/bold green]")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load_chunks() -> list[dict]:
    """Load saved chunks or ingest if not available."""
    chunks_path = ROOT_DIR / "labs" / ".chunks_lab33.json"
    if chunks_path.exists():
        with open(chunks_path, "r", encoding="utf-8") as f:
            return json.load(f)
    else:
        console.print("[yellow]Chunks not found. Running ingestion...[/yellow]")
        documents = load_pdfs(ROOT_DIR)
        chunks = split_documents(documents, chunk_size=512, chunk_overlap=50)
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        return chunks


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app()
