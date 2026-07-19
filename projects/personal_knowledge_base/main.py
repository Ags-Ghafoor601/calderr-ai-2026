#!/usr/bin/env python3
"""
Personal Knowledge Base — Project 3-I-A
=========================================
A personal Q&A system that ingests documents (PDF, TXT, MD), stores them in
ChromaDB with sentence-transformer embeddings, and answers questions with
source citations using Groq LLM.

Features:
  • Multi-format document loading (PDF, TXT, MD)
  • Configurable text splitting with chunk size tuning
  • ChromaDB vector storage with metadata (source, page, date)
  • Source filtering — query specific document collections
  • Streaming LLM responses via Groq
  • Rich CLI interface with beautiful terminal output
  • 15 Q&A demonstration examples

Usage:
    python main.py ingest ./sample_docs                     # Ingest documents
    python main.py ask "What is transfer learning?"          # Ask a question
    python main.py ask "What is NLP?" --source "03_*"       # Filter by source
    python main.py list-sources                              # List all sources
    python main.py stats                                     # Show collection stats
    python main.py demo                                      # Run 15 Q&A demo
"""

import io
import json
import os
import sys
import time
from datetime import datetime
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
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn
from rich.markdown import Markdown
from rich import box

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT_DIR.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

console = Console()
app = typer.Typer(
    name="knowledge-base",
    help="📚 Personal Knowledge Base — RAG-powered Q&A over your documents",
    add_completion=False,
)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
CHROMA_DIR = ROOT_DIR / ".chromadb"
COLLECTION_NAME = "personal_kb"


# ---------------------------------------------------------------------------
# Document Loading
# ---------------------------------------------------------------------------
def load_documents(docs_path: str) -> list[dict]:
    """Load documents from a directory supporting PDF, TXT, and MD formats."""
    docs_dir = Path(docs_path)
    if not docs_dir.exists():
        console.print(f"[red]Directory not found: {docs_dir}[/red]")
        raise typer.Exit(1)

    documents = []
    supported = {".pdf", ".txt", ".md", ".markdown", ".rst"}

    files = [f for f in docs_dir.rglob("*") if f.suffix.lower() in supported]
    if not files:
        console.print(f"[red]No supported files found in {docs_dir}[/red]")
        raise typer.Exit(1)

    with Progress(
        SpinnerColumn(),
        TextColumn("[bold cyan]Loading documents..."),
        BarColumn(),
        TextColumn("{task.completed}/{task.total}"),
        console=console,
    ) as progress:
        task = progress.add_task("load", total=len(files))

        for fpath in sorted(files):
            try:
                if fpath.suffix.lower() == ".pdf":
                    docs = _load_pdf(fpath)
                else:
                    docs = _load_text(fpath)
                documents.extend(docs)
            except Exception as e:
                console.print(f"  [yellow]⚠ Skipped {fpath.name}: {e}[/yellow]")
            progress.advance(task)

    console.print(
        f"  [green]✓[/green] Loaded [bold]{len(documents)}[/bold] document sections "
        f"from [bold]{len(files)}[/bold] files"
    )
    return documents


def _load_pdf(path: Path) -> list[dict]:
    """Load a PDF file using PyMuPDF."""
    import fitz

    docs = []
    pdf = fitz.open(str(path))
    for page_num, page in enumerate(pdf, 1):
        text = page.get_text().strip()
        if text and len(text) > 30:
            docs.append({
                "text": text,
                "metadata": {
                    "source": path.name,
                    "page": page_num,
                    "format": "pdf",
                    "ingested_at": datetime.now().isoformat(),
                },
            })
    pdf.close()
    return docs


def _load_text(path: Path) -> list[dict]:
    """Load a text/markdown file."""
    text = path.read_text(encoding="utf-8", errors="ignore").strip()
    if not text or len(text) < 30:
        return []

    return [{
        "text": text,
        "metadata": {
            "source": path.name,
            "page": 1,
            "format": path.suffix.lstrip("."),
            "ingested_at": datetime.now().isoformat(),
        },
    }]


# ---------------------------------------------------------------------------
# Text Splitting
# ---------------------------------------------------------------------------
def split_documents(
    documents: list[dict],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> list[dict]:
    """Split documents into chunks using recursive character splitting."""
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
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
                    "chunk_total": len(splits),
                },
            })

    console.print(
        f"  [green]✓[/green] Split into [bold]{len(chunks)}[/bold] chunks "
        f"(size={chunk_size}, overlap={chunk_overlap})"
    )
    return chunks


# ---------------------------------------------------------------------------
# ChromaDB Storage
# ---------------------------------------------------------------------------
def get_chromadb_collection(create: bool = False):
    """Get or create the ChromaDB collection."""
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    if create:
        try:
            client.delete_collection(COLLECTION_NAME)
        except Exception:
            pass
        return client.create_collection(
            name=COLLECTION_NAME,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )

    return client.get_collection(name=COLLECTION_NAME, embedding_function=ef)


def store_chunks(chunks: list[dict]):
    """Store document chunks in ChromaDB."""
    collection = get_chromadb_collection(create=True)

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
            batch = chunks[i:i + batch_size]
            collection.add(
                ids=[f"doc_{i + j}" for j in range(len(batch))],
                documents=[c["text"] for c in batch],
                metadatas=[c["metadata"] for c in batch],
            )
            progress.advance(task, len(batch))

    console.print(
        f"  [green]✓[/green] Stored [bold]{collection.count()}[/bold] chunks "
        f"in collection '{COLLECTION_NAME}'"
    )
    return collection


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------
def retrieve(query: str, top_k: int = 5, source_filter: str | None = None) -> list[dict]:
    """Retrieve relevant chunks from ChromaDB with optional source filtering."""
    collection = get_chromadb_collection()

    where_filter = None
    if source_filter:
        # Support glob-like patterns with *
        if "*" in source_filter:
            prefix = source_filter.replace("*", "")
            # ChromaDB doesn't support glob; we'll filter post-retrieval
            # Retrieve the whole collection to prevent the "post-filtering problem"
            results = collection.query(
                query_texts=[query],
                n_results=collection.count(),  # Retrieve all to filter safely
            )
        else:
            where_filter = {"source": source_filter}
            results = collection.query(
                query_texts=[query],
                n_results=top_k,
                where=where_filter,
            )
    else:
        results = collection.query(query_texts=[query], n_results=top_k)

    retrieved = []
    for i in range(len(results["documents"][0])):
        meta = results["metadatas"][0][i]

        # Apply glob filtering if needed
        if source_filter and "*" in source_filter:
            prefix = source_filter.replace("*", "")
            if not meta.get("source", "").startswith(prefix):
                continue

        retrieved.append({
            "text": results["documents"][0][i],
            "metadata": meta,
            "distance": results["distances"][0][i] if results.get("distances") else 0,
        })

        if len(retrieved) >= top_k:
            break

    return retrieved


# ---------------------------------------------------------------------------
# Answer Generation (with streaming)
# ---------------------------------------------------------------------------
def generate_answer(query: str, chunks: list[dict], stream: bool = True) -> str:
    """Generate an answer using Groq LLM with retrieved context."""
    from groq import Groq

    context = "\n\n---\n\n".join(
        f"[Source: {c['metadata'].get('source', '?')}, "
        f"Page {c['metadata'].get('page', '?')}]\n{c['text']}"
        for c in chunks
    )

    prompt = f"""You are a knowledgeable assistant for a personal knowledge base.
Answer the question based ONLY on the provided context.
If the context doesn't contain enough information, say so clearly.
Always cite the source document name and page number in your answer.
Format your answer clearly with proper structure.

Context:
{context}

Question: {query}

Answer:"""

    client = Groq(api_key=GROQ_API_KEY)

    if stream:
        full_response = ""
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
            stream=True,
        )

        console.print("\n[bold green]💡 Answer:[/bold green]")
        for chunk in response:
            delta = chunk.choices[0].delta.content or ""
            full_response += delta
            console.print(delta, end="")
        console.print()  # Newline after streaming
        return full_response
    else:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# Display Helpers
# ---------------------------------------------------------------------------
def display_sources(chunks: list[dict]):
    """Display source chunks in a formatted table."""
    table = Table(
        title="📄 Retrieved Sources",
        box=box.ROUNDED,
        border_style="dim cyan",
        show_lines=True,
    )
    table.add_column("#", style="bold cyan", width=3)
    table.add_column("Source", style="yellow", width=30)
    table.add_column("Page", style="green", width=6)
    table.add_column("Similarity", style="magenta", width=10)
    table.add_column("Preview", style="white", ratio=1)

    for i, c in enumerate(chunks, 1):
        sim = 1 - c.get("distance", 0)
        sim_color = "green" if sim > 0.6 else "yellow" if sim > 0.4 else "red"
        preview = c["text"][:100].replace("\n", " ") + "..."
        table.add_row(
            str(i),
            c["metadata"].get("source", "?"),
            str(c["metadata"].get("page", "?")),
            f"[{sim_color}]{sim:.3f}[/{sim_color}]",
            preview,
        )

    console.print(table)


# ---------------------------------------------------------------------------
# Demo Q&A examples
# ---------------------------------------------------------------------------
DEMO_QA = [
    "What is machine learning and what are its main types?",
    "Explain how neural networks learn through backpropagation.",
    "What are the core tasks in Natural Language Processing?",
    "How do convolutional neural networks work for image classification?",
    "What is the difference between Q-Learning and Policy Gradient methods?",
    "Explain the concept of transfer learning and its benefits.",
    "What are the key technologies in Generative AI?",
    "How are Large Language Models trained?",
    "What are embeddings and why are they useful?",
    "Explain the attention mechanism in transformers.",
    "What is LoRA and how does it enable efficient fine-tuning?",
    "What are the main prompt engineering techniques?",
    "How does RAG work and why is it important?",
    "Compare ChromaDB, FAISS, and Qdrant vector databases.",
    "What are the key practices in MLOps?",
]


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------
@app.command()
def ingest(
    docs_path: str = typer.Argument("./sample_docs", help="Path to documents directory"),
    chunk_size: int = typer.Option(512, "--chunk-size", "-c"),
    chunk_overlap: int = typer.Option(50, "--overlap", "-o"),
):
    """Ingest documents from a directory into the knowledge base."""
    console.print(Panel(
        "[bold]📚 Document Ingestion[/bold]\n"
        f"Source: {docs_path}\n"
        f"Chunk size: {chunk_size} | Overlap: {chunk_overlap}",
        style="bright_blue",
        border_style="bright_blue",
    ))

    documents = load_documents(docs_path)
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    store_chunks(chunks)

    console.print("\n[bold green]✅ Knowledge base built successfully![/bold green]")


@app.command()
def ask(
    question: str = typer.Argument(..., help="Your question"),
    top_k: int = typer.Option(5, "--top-k", "-k"),
    source: str = typer.Option(None, "--source", "-s", help="Filter by source file (supports *)"),
    no_stream: bool = typer.Option(False, "--no-stream"),
):
    """Ask a question and get an answer from your knowledge base."""
    console.print(Panel(
        f"[bold cyan]❓ Question:[/bold cyan] {question}"
        + (f"\n[dim]Source filter: {source}[/dim]" if source else ""),
        border_style="bright_blue",
    ))

    try:
        chunks = retrieve(question, top_k=top_k, source_filter=source)
    except Exception:
        console.print("[red]Knowledge base not found. Run 'ingest' first.[/red]")
        raise typer.Exit(1)

    if not chunks:
        console.print("[yellow]No relevant documents found.[/yellow]")
        raise typer.Exit(0)

    display_sources(chunks)

    answer = generate_answer(question, chunks, stream=not no_stream)

    if no_stream:
        console.print(Panel(
            f"[bold green]Answer:[/bold green]\n{answer}",
            border_style="green",
            title="💡 Knowledge Base Response",
        ))


@app.command("list-sources")
def list_sources():
    """List all unique document sources in the knowledge base."""
    try:
        collection = get_chromadb_collection()
    except Exception:
        console.print("[red]Knowledge base not found. Run 'ingest' first.[/red]")
        raise typer.Exit(1)

    # Get all metadata
    all_data = collection.get(include=["metadatas"])
    sources = {}
    for meta in all_data["metadatas"]:
        src = meta.get("source", "unknown")
        if src not in sources:
            sources[src] = {"count": 0, "format": meta.get("format", "?")}
        sources[src]["count"] += 1

    table = Table(
        title="📁 Knowledge Base Sources",
        box=box.ROUNDED,
        border_style="bright_blue",
    )
    table.add_column("Source File", style="cyan")
    table.add_column("Format", style="yellow")
    table.add_column("Chunks", style="green", justify="center")

    for src, info in sorted(sources.items()):
        table.add_row(src, info["format"], str(info["count"]))

    console.print(table)
    console.print(f"\n  [dim]Total: {len(sources)} sources, {collection.count()} chunks[/dim]")


@app.command()
def stats():
    """Show knowledge base statistics."""
    try:
        collection = get_chromadb_collection()
    except Exception:
        console.print("[red]Knowledge base not found. Run 'ingest' first.[/red]")
        raise typer.Exit(1)

    count = collection.count()
    all_data = collection.get(include=["metadatas"])

    sources = set()
    formats = set()
    for meta in all_data["metadatas"]:
        sources.add(meta.get("source", "?"))
        formats.add(meta.get("format", "?"))

    console.print(Panel(
        f"[bold]📊 Knowledge Base Statistics[/bold]\n\n"
        f"  Collection: [cyan]{COLLECTION_NAME}[/cyan]\n"
        f"  Total chunks: [green]{count}[/green]\n"
        f"  Unique sources: [yellow]{len(sources)}[/yellow]\n"
        f"  Formats: [magenta]{', '.join(formats)}[/magenta]\n"
        f"  Storage: [dim]{CHROMA_DIR}[/dim]",
        border_style="bright_blue",
    ))


@app.command()
def demo():
    """Run the full 15-question Q&A demonstration."""
    console.print(Panel(
        "[bold]🚀 Personal Knowledge Base — Full Demo[/bold]\n"
        f"Running {len(DEMO_QA)} Q&A examples",
        style="bright_magenta",
    ))

    try:
        collection = get_chromadb_collection()
    except Exception:
        console.print("[yellow]Knowledge base not built. Ingesting sample docs first...[/yellow]")
        docs = load_documents(str(ROOT_DIR / "sample_docs"))
        chunks = split_documents(docs)
        store_chunks(chunks)

    results = []
    for i, question in enumerate(DEMO_QA, 1):
        console.print(f"\n{'='*70}")
        console.print(f"[bold cyan]Question {i}/{len(DEMO_QA)}:[/bold cyan] {question}")
        console.print(f"{'='*70}")

        try:
            chunks = retrieve(question, top_k=3)
            display_sources(chunks)
            answer = generate_answer(question, chunks, stream=False)
            console.print(Panel(
                f"[green]{answer}[/green]",
                title="💡 Answer",
                border_style="green",
            ))
            results.append({"question": question, "answer": answer, "sources": len(chunks)})
        except Exception as e:
            console.print(f"[red]Error: {e}[/red]")
            results.append({"question": question, "error": str(e)})

        time.sleep(0.5)  # Rate limiting

    # Save results
    report_path = ROOT_DIR / "qa_examples.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    console.print(f"\n{'='*70}")
    console.print(f"[bold green]✅ Demo complete! {len(results)} Q&A pairs saved → {report_path}[/bold green]")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app()
