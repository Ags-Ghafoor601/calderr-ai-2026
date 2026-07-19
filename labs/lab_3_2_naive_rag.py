#!/usr/bin/env python3
"""
Lab 3.2 — Naive RAG Pipeline
==============================
Builds a complete RAG pipeline over PDF documents:
  load → split → embed → store in ChromaDB → retrieve → generate (Groq LLM)

Features:
  • Uses CalderR internship PDFs as source documents
  • Chunk size experiments: 256, 512, 1024 tokens
  • Metadata storage (source file, page number)
  • Retrieval accuracy evaluation with 20 Q&A pairs
  • Experiments with k=3, k=5, k=10 retrieved chunks

Usage:
    python lab_3_2_naive_rag.py ingest            # Ingest PDFs into ChromaDB
    python lab_3_2_naive_rag.py query "question"   # Ask a question
    python lab_3_2_naive_rag.py experiment          # Run chunk size experiments
    python lab_3_2_naive_rag.py evaluate            # Evaluate with 20 Q&A pairs
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
    name="naive-rag",
    help="📚 Naive RAG Pipeline — Lab 3.2",
    add_completion=False,
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_DIR = ROOT_DIR / "labs" / ".chromadb_lab32"
PDF_DIR = ROOT_DIR  # PDFs are in the root directory

# Default chunk parameters
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 50
DEFAULT_TOP_K = 5

# ---------------------------------------------------------------------------
# 20 Q&A pairs for evaluation (based on CalderR internship content)
# ---------------------------------------------------------------------------
QA_PAIRS = [
    {
        "question": "What is the primary theme of Week 1 of the CalderR internship?",
        "ground_truth": "Week 1 focuses on LLM fundamentals, prompt engineering, and building the first AI applications.",
    },
    {
        "question": "What programming language is the primary stack for the internship?",
        "ground_truth": "Python is the primary programming language used throughout the internship.",
    },
    {
        "question": "What LLM provider does the internship use for API calls?",
        "ground_truth": "The internship uses Groq as the LLM provider for fast inference.",
    },
    {
        "question": "What is the purpose of embeddings in RAG systems?",
        "ground_truth": "Embeddings represent text as dense numerical vectors that capture semantic meaning, enabling similarity search.",
    },
    {
        "question": "What vector database is primarily used in Week 3?",
        "ground_truth": "ChromaDB is the primary vector database used in Week 3.",
    },
    {
        "question": "What framework is used for RAG evaluation?",
        "ground_truth": "RAGAS is the framework used for evaluating RAG systems.",
    },
    {
        "question": "How many hours per week is the total commitment for the internship?",
        "ground_truth": "The total commitment is 20 hours per week, approximately 4 hours per day for 5 days.",
    },
    {
        "question": "What is hybrid search in the context of RAG?",
        "ground_truth": "Hybrid search combines BM25 keyword search with semantic vector search to improve retrieval quality.",
    },
    {
        "question": "What is a cross-encoder re-ranker?",
        "ground_truth": "A cross-encoder re-ranker scores query-document pairs jointly to re-order retrieved results for better relevance.",
    },
    {
        "question": "What sentence-transformer models are compared in Lab 3.1?",
        "ground_truth": "Lab 3.1 compares all-MiniLM-L6-v2 and BAAI/bge-small-en-v1.5 embedding models.",
    },
    {
        "question": "What are the chunk sizes tested in Lab 3.2?",
        "ground_truth": "Lab 3.2 tests chunk sizes of 256, 512, and 1024 tokens.",
    },
    {
        "question": "What are the three RAGAS metrics mentioned in the syllabus?",
        "ground_truth": "The three RAGAS metrics are faithfulness, answer relevancy, and context precision.",
    },
    {
        "question": "What is the deliverable for the Friday standup?",
        "ground_truth": "The deliverables include one Intermediate Project and one Production Project, a live demo, architecture review, and RAGAS evaluation results.",
    },
    {
        "question": "What is multi-tenancy in the Enterprise Document Intelligence Platform?",
        "ground_truth": "Multi-tenancy means each customer's documents are isolated in separate ChromaDB namespaces.",
    },
    {
        "question": "What is the purpose of the weekly standup?",
        "ground_truth": "The weekly standup includes a live demo, technical question, architecture review, code review, and discussion of blockers.",
    },
    {
        "question": "What are the categories of projects for Week 3?",
        "ground_truth": "Week 3 has two categories: Intermediate projects (Category 1) and Production projects (Category 2).",
    },
    {
        "question": "What does the Personal Knowledge Base project involve?",
        "ground_truth": "Building a personal Q&A system from documents supporting multi-document ingestion, filtering by source, and streaming responses.",
    },
    {
        "question": "What UI framework is suggested for the Product Manual Assistant?",
        "ground_truth": "Streamlit is suggested as the UI framework for the Product Manual Assistant.",
    },
    {
        "question": "What does FAISS stand for and what is it used for?",
        "ground_truth": "FAISS is Facebook AI Similarity Search, used as an in-process vector index for efficient similarity search.",
    },
    {
        "question": "What is the evaluation criteria for the Enterprise platform project?",
        "ground_truth": "Multi-tenancy must work correctly, RAGAS scores should be greater than 0.7, Docker compose must work, and the API must be fully documented.",
    },
]


# ---------------------------------------------------------------------------
# PDF Loading
# ---------------------------------------------------------------------------
def load_pdfs(pdf_dir: Path) -> list[dict]:
    """Load all CalderR PDFs and return a list of document dicts."""
    import fitz  # PyMuPDF

    documents = []
    pdf_files = sorted(pdf_dir.glob("CalderR_*.pdf"))

    if not pdf_files:
        console.print("[red]No CalderR PDF files found![/red]")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Loading PDFs..."),
        BarColumn(),
        TextColumn("{task.completed}/{task.total} files"),
        console=console,
    ) as progress:
        task = progress.add_task("load", total=len(pdf_files))
        for pdf_path in pdf_files:
            try:
                doc = fitz.open(str(pdf_path))
                for page_num, page in enumerate(doc, 1):
                    text = page.get_text().strip()
                    if text and len(text) > 50:  # Skip nearly-empty pages
                        documents.append({
                            "text": text,
                            "metadata": {
                                "source": pdf_path.name,
                                "page": page_num,
                                "total_pages": len(doc),
                            },
                        })
                doc.close()
            except Exception as e:
                console.print(f"  [yellow]⚠ Skipped {pdf_path.name}: {e}[/yellow]")
            progress.advance(task)

    console.print(f"  [green]✓[/green] Loaded [bold]{len(documents)}[/bold] pages from {len(pdf_files)} PDFs")
    return documents


# ---------------------------------------------------------------------------
# Text Splitting
# ---------------------------------------------------------------------------
def split_documents(documents: list[dict], chunk_size: int = 512, chunk_overlap: int = 50) -> list[dict]:
    """Split documents into chunks using recursive character splitting."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = []
    for doc in documents:
        splits = splitter.split_text(doc["text"])
        for i, chunk_text in enumerate(splits):
            chunks.append({
                "text": chunk_text,
                "metadata": {
                    **doc["metadata"],
                    "chunk_index": i,
                    "chunk_size_setting": chunk_size,
                },
            })

    console.print(
        f"  [green]✓[/green] Split into [bold]{len(chunks)}[/bold] chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


# ---------------------------------------------------------------------------
# Embedding & ChromaDB Storage
# ---------------------------------------------------------------------------
def store_in_chromadb(
    chunks: list[dict],
    collection_name: str = "naive_rag",
    chroma_dir: Path = CHROMA_DIR,
) -> "chromadb.Collection":
    """Embed chunks and store in ChromaDB with metadata."""
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")

    client = chromadb.PersistentClient(path=str(chroma_dir))

    # Delete existing collection if present, then create fresh
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass

    collection = client.create_collection(
        name=collection_name,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )

    # Batch add (ChromaDB limit is ~5000 per batch)
    batch_size = 500
    with Progress(
        SpinnerColumn(),
        TextColumn("[bold green]Storing in ChromaDB..."),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("store", total=len(chunks))
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i : i + batch_size]
            collection.add(
                ids=[f"chunk_{i + j}" for j in range(len(batch))],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
            )
            progress.advance(task, len(batch))

    console.print(
        f"  [green]✓[/green] Stored [bold]{collection.count()}[/bold] chunks in "
        f"collection '{collection_name}'"
    )
    return collection


def get_collection(
    collection_name: str = "naive_rag",
    chroma_dir: Path = CHROMA_DIR,
):
    """Retrieve an existing ChromaDB collection."""
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(chroma_dir))
    return client.get_collection(name=collection_name, embedding_function=ef)


# ---------------------------------------------------------------------------
# Retrieval & Generation
# ---------------------------------------------------------------------------
def retrieve(collection, query: str, top_k: int = 5) -> list[dict]:
    """Retrieve the top-k most relevant chunks for a query."""
    results = collection.query(query_texts=[query], n_results=top_k)
    retrieved = []
    for i in range(len(results["documents"][0])):
        retrieved.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i] if results.get("distances") else None,
        })
    return retrieved


def generate_answer(query: str, context_chunks: list[dict]) -> str:
    """Generate an answer using Groq LLM with retrieved context."""
    from groq import Groq

    context = "\n\n---\n\n".join(
        f"[Source: {c['metadata'].get('source', 'unknown')}, Page {c['metadata'].get('page', '?')}]\n{c['text']}"
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
# Display helpers
# ---------------------------------------------------------------------------
def display_answer(query: str, answer: str, chunks: list[dict]):
    """Display the RAG answer with source citations."""
    console.print()
    console.print(Panel(
        f"[bold cyan]Question:[/bold cyan] {query}",
        border_style="bright_blue",
    ))
    console.print(Panel(
        f"[bold green]Answer:[/bold green]\n{answer}",
        border_style="green",
        title="💡 RAG Response",
    ))

    # Sources table
    table = Table(
        title="📄 Retrieved Sources",
        box=box.ROUNDED,
        border_style="dim",
        show_lines=True,
    )
    table.add_column("#", style="bold cyan", width=3)
    table.add_column("Source", style="yellow", width=25)
    table.add_column("Page", style="green", width=6)
    table.add_column("Chunk Preview", style="white", ratio=1)

    for i, c in enumerate(chunks, 1):
        preview = c["text"][:100].replace("\n", " ") + "..."
        table.add_row(
            str(i),
            c["metadata"].get("source", "?"),
            str(c["metadata"].get("page", "?")),
            preview,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------
@app.command()
def ingest(
    chunk_size: int = typer.Option(DEFAULT_CHUNK_SIZE, "--chunk-size", "-c"),
    chunk_overlap: int = typer.Option(DEFAULT_CHUNK_OVERLAP, "--overlap", "-o"),
    collection: str = typer.Option("naive_rag", "--collection"),
):
    """Ingest CalderR PDFs into ChromaDB."""
    console.print(Panel(
        "[bold]📚 PDF Ingestion Pipeline[/bold]\n"
        f"Chunk size: {chunk_size} | Overlap: {chunk_overlap} | Collection: {collection}",
        style="bright_blue",
    ))

    documents = load_pdfs(PDF_DIR)
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store_in_chromadb(chunks, collection_name=collection)

    console.print("\n[bold green]✅ Ingestion complete![/bold green]")


@app.command()
def query(
    question: str = typer.Argument(..., help="Question to ask the RAG system"),
    top_k: int = typer.Option(DEFAULT_TOP_K, "--top-k", "-k"),
    collection: str = typer.Option("naive_rag", "--collection"),
):
    """Ask a question and get a RAG-generated answer."""
    try:
        coll = get_collection(collection)
    except Exception:
        console.print("[red]Collection not found. Run 'ingest' first.[/red]")
        raise typer.Exit(1)

    chunks = retrieve(coll, question, top_k=top_k)
    answer = generate_answer(question, chunks)
    display_answer(question, answer, chunks)


@app.command()
def experiment():
    """Run chunk size experiments (256, 512, 1024) and compare retrieval quality."""
    console.print(Panel(
        "[bold]🔬 Chunk Size Experiment[/bold]\n"
        "Testing chunk sizes: 256, 512, 1024\n"
        "Evaluating retrieval quality across all Q&A pairs",
        style="bright_magenta",
    ))

    documents = load_pdfs(PDF_DIR)
    chunk_sizes = [256, 512, 1024]
    results = {}

    for cs in chunk_sizes:
        console.print(f"\n{'='*60}")
        console.print(f"[bold]Chunk size: {cs}[/bold]")
        console.print(f"{'='*60}")

        collection_name = f"experiment_cs{cs}"
        chunks = split_documents(documents, chunk_size=cs, chunk_overlap=cs // 10)
        coll = store_in_chromadb(chunks, collection_name=collection_name)

        # Evaluate with 5 sample queries
        test_queries = [qa["question"] for qa in QA_PAIRS[:5]]
        retrieval_scores = []

        for q in test_queries:
            retrieved = retrieve(coll, q, top_k=5)
            # Simple relevance heuristic: check if any chunk contains key terms
            query_terms = set(q.lower().split())
            for chunk in retrieved:
                chunk_terms = set(chunk["text"].lower().split())
                overlap = len(query_terms & chunk_terms)
                retrieval_scores.append(overlap / max(len(query_terms), 1))

        avg_score = np.mean(retrieval_scores) if retrieval_scores else 0
        results[cs] = {
            "num_chunks": coll.count(),
            "avg_retrieval_score": avg_score,
        }

        console.print(
            f"  Chunks: {coll.count()} | Avg Retrieval Score: {avg_score:.3f}"
        )

    # Summary table
    console.print(f"\n{'='*60}")
    table = Table(
        title="📊 Chunk Size Experiment Results",
        box=box.DOUBLE_EDGE,
        title_style="bold magenta",
        border_style="bright_blue",
    )
    table.add_column("Chunk Size", style="bold cyan", justify="center")
    table.add_column("Total Chunks", style="green", justify="center")
    table.add_column("Avg Retrieval Score", style="yellow", justify="center")

    for cs, data in results.items():
        table.add_row(
            str(cs),
            str(data["num_chunks"]),
            f"{data['avg_retrieval_score']:.4f}",
        )

    console.print(table)
    console.print("\n[bold green]✅ Experiment complete![/bold green]")

    # Save results
    report_path = ROOT_DIR / "labs" / "lab_3_2_experiment_results.json"
    with open(report_path, "w") as f:
        json.dump(results, f, indent=2)
    console.print(f"  [dim]Results saved → {report_path}[/dim]")


@app.command()
def evaluate():
    """Evaluate the RAG pipeline on 20 Q&A pairs and report accuracy."""
    console.print(Panel(
        "[bold]📝 RAG Evaluation[/bold]\n"
        f"Running {len(QA_PAIRS)} Q&A pairs with k=3, k=5, k=10",
        style="bright_magenta",
    ))

    # Ensure collection exists
    try:
        coll = get_collection("naive_rag")
    except Exception:
        console.print("[yellow]Collection not found. Ingesting first...[/yellow]")
        documents = load_pdfs(PDF_DIR)
        chunks = split_documents(documents, chunk_size=512, chunk_overlap=50)
        coll = store_in_chromadb(chunks, collection_name="naive_rag")

    k_values = [3, 5, 10]
    all_results = {}

    for k in k_values:
        console.print(f"\n[bold]Testing with k={k}[/bold]")
        eval_results = []

        with Progress(
            SpinnerColumn(),
            TextColumn(f"[cyan]Evaluating k={k}..."),
            BarColumn(),
            TextColumn("{task.completed}/{task.total}"),
            console=console,
        ) as progress:
            task = progress.add_task("eval", total=len(QA_PAIRS))

            for qa in QA_PAIRS:
                try:
                    chunks = retrieve(coll, qa["question"], top_k=k)
                    answer = generate_answer(qa["question"], chunks)

                    # Simple overlap-based scoring
                    gt_terms = set(qa["ground_truth"].lower().split())
                    ans_terms = set(answer.lower().split())
                    overlap = len(gt_terms & ans_terms)
                    score = overlap / max(len(gt_terms), 1)

                    eval_results.append({
                        "question": qa["question"],
                        "ground_truth": qa["ground_truth"],
                        "answer": answer,
                        "score": score,
                        "num_chunks": len(chunks),
                    })
                except Exception as e:
                    eval_results.append({
                        "question": qa["question"],
                        "score": 0,
                        "error": str(e),
                    })
                progress.advance(task)

                # Rate limiting for Groq API
                time.sleep(0.5)

        avg_score = np.mean([r["score"] for r in eval_results])
        all_results[k] = {"avg_score": avg_score, "details": eval_results}
        console.print(f"  Average Score (k={k}): [bold]{avg_score:.3f}[/bold]")

    # Summary table
    table = Table(
        title="📊 Evaluation Results by k",
        box=box.DOUBLE_EDGE,
        title_style="bold magenta",
        border_style="bright_blue",
    )
    table.add_column("k", style="bold cyan", justify="center")
    table.add_column("Avg Score", style="green", justify="center")

    for k, data in all_results.items():
        score_color = "green" if data["avg_score"] > 0.4 else "yellow"
        table.add_row(str(k), f"[{score_color}]{data['avg_score']:.4f}[/{score_color}]")

    console.print(table)

    # Save evaluation report
    report_path = ROOT_DIR / "labs" / "lab_3_2_evaluation_report.json"
    serializable = {
        str(k): {"avg_score": v["avg_score"], "num_questions": len(v["details"])}
        for k, v in all_results.items()
    }
    with open(report_path, "w") as f:
        json.dump(serializable, f, indent=2)
    console.print(f"\n  [dim]Report saved → {report_path}[/dim]")
    console.print("[bold green]✅ Evaluation complete![/bold green]")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app()
