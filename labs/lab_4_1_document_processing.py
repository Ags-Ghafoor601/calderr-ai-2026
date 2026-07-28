#!/usr/bin/env python3
"""
CalderR Internship – Week 4, Lab 4.1
======================================
Document Processing Graph — LangGraph Workflows with Conditional Routing

WHAT THIS LAB BUILDS:
---------------------
A document processing pipeline using LangGraph's StateGraph that:
  • Loads documents from raw text input
  • Validates document structure (length, format checks)
  • Conditionally splits oversized documents before chunking
  • Chunks documents into manageable pieces with overlap
  • Generates embeddings for each chunk (simulated)
  • Confirms processing with a summary report

WHAT THIS TEACHES YOU:
----------------------
  • LangGraph StateGraph fundamentals (nodes, edges, conditional edges)
  • TypedDict-based state management with proper typing
  • Conditional routing based on document properties
  • Graph compilation and execution
  • Visualizing graph structure and execution flow

ARCHITECTURE:
                    ┌──────────┐
                    │   load   │
                    └────┬─────┘
                         │
                    ┌────▼─────┐
                    │ validate │
                    └────┬─────┘
                         │
                   ┌─────▼──────┐
                   │  oversized? │
                   └──┬──────┬──┘
              yes ┌───┘      └───┐ no
             ┌────▼────┐   ┌────▼────┐
             │  split   │   │  chunk  │
             └────┬────┘   └────┬────┘
                  │             │
             ┌────▼────┐       │
             │  chunk  │       │
             └────┬────┘       │
                  └──────┬─────┘
                    ┌────▼────┐
                    │  embed  │
                    └────┬────┘
                    ┌────▼─────┐
                    │ confirm  │
                    └──────────┘

Run:
    python labs/lab_4_1_document_processing.py demo
    python labs/lab_4_1_document_processing.py process "Your document text here"
    python labs/lab_4_1_document_processing.py process-file path/to/file.txt
    python labs/lab_4_1_document_processing.py graph
"""

import io
import os
import sys
import time
import math
import hashlib
from pathlib import Path
from typing import Optional, Annotated
from operator import add

# Fix Windows console encoding for Rich Unicode output
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import typer
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.tree import Tree
from rich.text import Text
from rich.rule import Rule
from rich.columns import Columns
from rich import box

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, END

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

console = Console()
app = typer.Typer(
    name="doc-processing",
    help="📄 Document Processing Graph — Lab 4.1",
    add_completion=False,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MAX_DOC_LENGTH = 2000          # Characters threshold for "oversized"
CHUNK_SIZE = 500               # Characters per chunk
CHUNK_OVERLAP = 50             # Overlap between chunks
EMBEDDING_DIM = 384            # Simulated embedding dimension
SPLIT_PARTS = 3                # How many parts to split oversized docs into


# ---------------------------------------------------------------------------
# State Schema (TypedDict)
# ---------------------------------------------------------------------------
class DocumentState(TypedDict):
    """Graph state for document processing pipeline.

    Each field tracks a different aspect of the processing:
      - raw_text: The original input text
      - document_id: Unique hash-based identifier
      - is_valid: Whether the document passed validation
      - validation_errors: List of validation issues found
      - is_oversized: Whether the document exceeds MAX_DOC_LENGTH
      - parts: Sub-documents after splitting (if oversized)
      - chunks: Final text chunks ready for embedding
      - embeddings: Simulated vector embeddings
      - processing_log: Annotated list that accumulates log entries
      - total_processing_time: Seconds elapsed during processing
    """
    raw_text: str
    document_id: str
    is_valid: bool
    validation_errors: list[str]
    is_oversized: bool
    parts: list[str]
    chunks: list[str]
    embeddings: list[list[float]]
    processing_log: Annotated[list[str], add]
    total_processing_time: float


# ---------------------------------------------------------------------------
# Node Functions
# ---------------------------------------------------------------------------

def load_document(state: DocumentState) -> dict:
    """Load and register the raw document — assigns a unique ID."""
    start = time.time()
    raw = state["raw_text"]

    # Generate a hash-based document ID
    doc_id = hashlib.sha256(raw.encode()).hexdigest()[:12]

    elapsed = time.time() - start
    return {
        "document_id": doc_id,
        "processing_log": [
            f"[LOAD] Document registered — ID: {doc_id}, "
            f"Length: {len(raw):,} chars, "
            f"Words: {len(raw.split()):,} ({elapsed:.3f}s)"
        ],
    }


def validate_document(state: DocumentState) -> dict:
    """Validate document structure — check for empty, too short, encoding issues."""
    start = time.time()
    raw = state["raw_text"]
    errors = []

    # Check: Empty or whitespace only
    if not raw or not raw.strip():
        errors.append("Document is empty or contains only whitespace")

    # Check: Minimum length
    if len(raw.strip()) < 20:
        errors.append(f"Document too short ({len(raw.strip())} chars, minimum 20)")

    # Check: Excessive non-printable characters
    non_printable = sum(1 for c in raw if not c.isprintable() and c not in "\n\r\t")
    if non_printable > len(raw) * 0.1:
        errors.append(
            f"Too many non-printable characters ({non_printable}/{len(raw)}, "
            f">{10}% threshold)"
        )

    # Check: Repeated content (degenerate input)
    if len(raw) > 100:
        sample = raw[:50]
        if raw.count(sample) > 3:
            errors.append("Suspicious repeated content detected")

    is_valid = len(errors) == 0
    is_oversized = len(raw) > MAX_DOC_LENGTH and is_valid

    elapsed = time.time() - start
    status = "✅ PASSED" if is_valid else f"❌ FAILED ({len(errors)} issues)"
    log = [
        f"[VALIDATE] {status} — "
        f"Oversized: {'Yes' if is_oversized else 'No'} "
        f"(threshold: {MAX_DOC_LENGTH:,} chars) ({elapsed:.3f}s)"
    ]

    return {
        "is_valid": is_valid,
        "validation_errors": errors,
        "is_oversized": is_oversized,
        "processing_log": log,
    }


def split_document(state: DocumentState) -> dict:
    """Split oversized document into roughly equal parts."""
    start = time.time()
    raw = state["raw_text"]

    # Split by sentences first for cleaner breaks
    sentences = []
    current = ""
    for char in raw:
        current += char
        if char in ".!?" and len(current.strip()) > 10:
            sentences.append(current.strip())
            current = ""
    if current.strip():
        sentences.append(current.strip())

    # Distribute sentences across parts
    if len(sentences) <= SPLIT_PARTS:
        parts = sentences
    else:
        per_part = math.ceil(len(sentences) / SPLIT_PARTS)
        parts = []
        for i in range(0, len(sentences), per_part):
            part = " ".join(sentences[i:i + per_part])
            parts.append(part)

    elapsed = time.time() - start
    part_lengths = [len(p) for p in parts]
    return {
        "parts": parts,
        "processing_log": [
            f"[SPLIT] Document split into {len(parts)} parts — "
            f"Part sizes: {part_lengths} chars ({elapsed:.3f}s)"
        ],
    }


def chunk_document(state: DocumentState) -> dict:
    """Chunk document (or parts) into smaller pieces with overlap."""
    start = time.time()

    # If we have parts from splitting, chunk each part
    # Otherwise, chunk the raw text directly
    sources = state.get("parts") or [state["raw_text"]]
    if not sources or sources == [""]:
        sources = [state["raw_text"]]

    all_chunks = []
    for source_text in sources:
        text = source_text.strip()
        if not text:
            continue

        # Sliding-window chunking with overlap
        pos = 0
        while pos < len(text):
            end = min(pos + CHUNK_SIZE, len(text))
            chunk = text[pos:end].strip()
            if chunk:
                all_chunks.append(chunk)
            pos += CHUNK_SIZE - CHUNK_OVERLAP

    elapsed = time.time() - start
    avg_len = sum(len(c) for c in all_chunks) / max(len(all_chunks), 1)
    return {
        "chunks": all_chunks,
        "processing_log": [
            f"[CHUNK] Created {len(all_chunks)} chunks — "
            f"Avg length: {avg_len:.0f} chars, "
            f"Overlap: {CHUNK_OVERLAP} chars ({elapsed:.3f}s)"
        ],
    }


def embed_chunks(state: DocumentState) -> dict:
    """Generate simulated embeddings for each chunk.

    In production, this would call sentence-transformers or an API.
    Here we generate deterministic pseudo-embeddings for demonstration.
    """
    start = time.time()
    chunks = state["chunks"]
    embeddings = []

    for chunk in chunks:
        # Deterministic pseudo-embedding based on chunk content hash
        seed = int(hashlib.md5(chunk.encode()).hexdigest(), 16) % (2**32)
        import random as _rng
        _rng.seed(seed)
        vec = [_rng.gauss(0, 1) for _ in range(EMBEDDING_DIM)]
        # L2-normalize
        norm = math.sqrt(sum(v * v for v in vec))
        vec = [v / norm for v in vec]
        embeddings.append(vec)

    elapsed = time.time() - start
    return {
        "embeddings": embeddings,
        "processing_log": [
            f"[EMBED] Generated {len(embeddings)} embeddings — "
            f"Dimension: {EMBEDDING_DIM}, "
            f"Normalized: Yes ({elapsed:.3f}s)"
        ],
    }


def confirm_processing(state: DocumentState) -> dict:
    """Final confirmation node — summarize processing results."""
    start = time.time()
    elapsed = time.time() - start

    total_chars = sum(len(c) for c in state["chunks"])
    return {
        "total_processing_time": elapsed,
        "processing_log": [
            f"[CONFIRM] ✅ Processing complete — "
            f"Document {state['document_id']}: "
            f"{len(state['chunks'])} chunks, "
            f"{len(state['embeddings'])} embeddings, "
            f"{total_chars:,} total chars ({elapsed:.3f}s)"
        ],
    }


def handle_invalid(state: DocumentState) -> dict:
    """Handle invalid documents — log errors and terminate."""
    errors = state.get("validation_errors", [])
    return {
        "processing_log": [
            f"[REJECTED] ❌ Document invalid — {len(errors)} error(s): "
            + "; ".join(errors)
        ],
    }


# ---------------------------------------------------------------------------
# Conditional Edge Functions
# ---------------------------------------------------------------------------

def route_after_validation(state: DocumentState) -> str:
    """Route based on validation result and document size."""
    if not state.get("is_valid", False):
        return "handle_invalid"
    if state.get("is_oversized", False):
        return "split_document"
    return "chunk_document"


# ---------------------------------------------------------------------------
# Graph Construction
# ---------------------------------------------------------------------------

def build_graph():
    """Build and compile the document processing graph."""
    graph = StateGraph(DocumentState)

    # Add nodes
    graph.add_node("load_document", load_document)
    graph.add_node("validate_document", validate_document)
    graph.add_node("split_document", split_document)
    graph.add_node("chunk_document", chunk_document)
    graph.add_node("embed_chunks", embed_chunks)
    graph.add_node("confirm_processing", confirm_processing)
    graph.add_node("handle_invalid", handle_invalid)

    # Set entry point
    graph.set_entry_point("load_document")

    # Add edges
    graph.add_edge("load_document", "validate_document")

    # Conditional edge: route based on validation + size
    graph.add_conditional_edges(
        "validate_document",
        route_after_validation,
        {
            "handle_invalid": "handle_invalid",
            "split_document": "split_document",
            "chunk_document": "chunk_document",
        },
    )

    # Split → Chunk → Embed → Confirm → END
    graph.add_edge("split_document", "chunk_document")
    graph.add_edge("chunk_document", "embed_chunks")
    graph.add_edge("embed_chunks", "confirm_processing")
    graph.add_edge("confirm_processing", END)
    graph.add_edge("handle_invalid", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Display Helpers
# ---------------------------------------------------------------------------

def display_result(result: dict):
    """Display processing results in a rich format."""
    console.print()

    # Header
    status = "✅ SUCCESS" if result.get("is_valid", False) else "❌ REJECTED"
    color = "green" if result.get("is_valid", False) else "red"
    console.print(Panel(
        f"[bold {color}]{status}[/] — Document [cyan]{result.get('document_id', 'N/A')}[/]",
        title="📄 Processing Result",
        border_style=color,
    ))

    # Processing log timeline
    log = result.get("processing_log", [])
    if log:
        tree = Tree("📋 [bold]Processing Timeline[/]")
        for entry in log:
            if "✅" in entry or "CONFIRM" in entry:
                tree.add(f"[green]{entry}[/]")
            elif "❌" in entry or "REJECTED" in entry:
                tree.add(f"[red]{entry}[/]")
            elif "SPLIT" in entry:
                tree.add(f"[yellow]{entry}[/]")
            else:
                tree.add(f"[cyan]{entry}[/]")
        console.print(tree)

    # Validation errors
    errors = result.get("validation_errors", [])
    if errors:
        console.print()
        err_panel = "\n".join(f"  ⚠️  {e}" for e in errors)
        console.print(Panel(err_panel, title="❌ Validation Errors", border_style="red"))

    # Stats table
    if result.get("is_valid", False):
        console.print()
        table = Table(title="📊 Processing Statistics", box=box.ROUNDED)
        table.add_column("Metric", style="bold cyan")
        table.add_column("Value", style="bold white")

        table.add_row("Document ID", result.get("document_id", "N/A"))
        table.add_row("Input Length", f"{len(result.get('raw_text', '')):,} chars")
        table.add_row("Oversized?", "Yes → Split" if result.get("is_oversized") else "No")
        if result.get("parts"):
            table.add_row("Parts Created", str(len(result["parts"])))
        table.add_row("Chunks Created", str(len(result.get("chunks", []))))
        table.add_row("Embeddings Generated", str(len(result.get("embeddings", []))))
        table.add_row("Embedding Dimension", str(EMBEDDING_DIM))
        table.add_row("Chunk Size / Overlap", f"{CHUNK_SIZE} / {CHUNK_OVERLAP} chars")

        console.print(table)

    # Chunk preview
    chunks = result.get("chunks", [])
    if chunks:
        console.print()
        preview_table = Table(
            title=f"📝 Chunk Preview (showing first {min(5, len(chunks))} of {len(chunks)})",
            box=box.SIMPLE_HEAVY,
        )
        preview_table.add_column("#", style="dim", width=4)
        preview_table.add_column("Content (first 100 chars)", style="white")
        preview_table.add_column("Length", style="cyan", justify="right")

        for i, chunk in enumerate(chunks[:5]):
            preview = chunk[:100].replace("\n", "↵ ") + ("…" if len(chunk) > 100 else "")
            preview_table.add_row(str(i + 1), preview, f"{len(chunk):,}")

        console.print(preview_table)


def display_graph_structure():
    """Display the graph structure as a visual tree."""
    console.print()
    tree = Tree("🔷 [bold cyan]Document Processing Graph[/]")

    load = tree.add("📥 [bold]load_document[/] — Register & assign ID")
    validate = load.add("🔍 [bold]validate_document[/] — Check structure")
    conditional = validate.add("⚡ [bold yellow]CONDITIONAL EDGE[/]")

    invalid = conditional.add("❌ [red]is_valid = False[/] → [bold]handle_invalid[/] → END")

    oversized = conditional.add("📏 [yellow]is_oversized = True[/]")
    split = oversized.add("✂️  [bold]split_document[/] — Break into parts")
    chunk1 = split.add("🧩 [bold]chunk_document[/] — Sliding-window chunks")
    embed1 = chunk1.add("🔢 [bold]embed_chunks[/] — Generate vectors")
    embed1.add("✅ [bold green]confirm_processing[/] → END")

    normal = conditional.add("📄 [green]is_oversized = False[/]")
    chunk2 = normal.add("🧩 [bold]chunk_document[/] — Sliding-window chunks")
    embed2 = chunk2.add("🔢 [bold]embed_chunks[/] — Generate vectors")
    embed2.add("✅ [bold green]confirm_processing[/] → END")

    console.print(Panel(tree, title="📐 Graph Architecture", border_style="blue"))

    # Configuration
    console.print()
    config_table = Table(title="⚙️  Configuration", box=box.ROUNDED)
    config_table.add_column("Parameter", style="bold cyan")
    config_table.add_column("Value", style="bold white")
    config_table.add_row("Max Document Length", f"{MAX_DOC_LENGTH:,} chars")
    config_table.add_row("Chunk Size", f"{CHUNK_SIZE} chars")
    config_table.add_row("Chunk Overlap", f"{CHUNK_OVERLAP} chars")
    config_table.add_row("Embedding Dimension", str(EMBEDDING_DIM))
    config_table.add_row("Split Parts (oversized)", str(SPLIT_PARTS))
    console.print(config_table)


# ---------------------------------------------------------------------------
# Sample Documents for Demo
# ---------------------------------------------------------------------------

SAMPLE_SHORT = (
    "Artificial intelligence has transformed modern software development. "
    "Machine learning models can now understand natural language, generate code, "
    "and even reason about complex problems. LangGraph provides a powerful "
    "framework for building stateful AI workflows with branching, looping, "
    "and human-in-the-loop capabilities. This enables developers to create "
    "sophisticated agents that go beyond simple linear chains."
)

SAMPLE_OVERSIZED = (
    "The history of artificial intelligence spans over seven decades of research, "
    "innovation, and breakthrough discoveries. "
    "In 1950, Alan Turing published his seminal paper 'Computing Machinery and "
    "Intelligence,' introducing the concept of the Turing Test as a measure of "
    "machine intelligence. This paper laid the philosophical groundwork for the "
    "entire field. "
    "The term 'Artificial Intelligence' was coined at the Dartmouth Conference in "
    "1956, organized by John McCarthy, Marvin Minsky, Nathaniel Rochester, and "
    "Claude Shannon. This conference is widely considered the birth of AI as a "
    "formal academic discipline. "
    "During the 1960s and 1970s, early AI research focused on symbolic reasoning "
    "and expert systems. Programs like ELIZA (1966) demonstrated simple natural "
    "language processing, while SHRDLU (1970) showed how computers could understand "
    "language in limited domains. Expert systems like MYCIN became practical tools "
    "for medical diagnosis. "
    "The 1980s saw the rise and fall of expert systems. While initially successful "
    "in narrow domains, these rule-based systems proved brittle and expensive to "
    "maintain. This led to the first 'AI Winter,' a period of reduced funding and "
    "interest in AI research. "
    "The neural network renaissance began in the late 1980s with backpropagation, "
    "but it wasn't until the 2000s that deep learning truly emerged. Key milestones "
    "include Geoffrey Hinton's deep belief networks (2006), the ImageNet competition "
    "victory by AlexNet (2012), and the development of attention mechanisms leading "
    "to the Transformer architecture (2017). "
    "The Transformer architecture, introduced in the paper 'Attention Is All You Need,' "
    "revolutionized natural language processing. It enabled models like BERT, GPT, "
    "and T5 that could understand and generate human language with unprecedented "
    "fluency and accuracy. "
    "Large Language Models (LLMs) like GPT-3 (2020) and GPT-4 (2023) demonstrated "
    "emergent capabilities including in-context learning, chain-of-thought reasoning, "
    "and tool use. These models can write code, solve math problems, and engage in "
    "complex multi-step reasoning. "
    "The development of agentic AI systems represents the latest frontier. Frameworks "
    "like LangChain and LangGraph enable developers to build AI agents that can use "
    "tools, maintain state across interactions, branch and loop through complex "
    "workflows, and collaborate with human operators through human-in-the-loop patterns. "
    "These agentic systems are being deployed in production for tasks ranging from "
    "customer support automation to scientific research assistance, code generation, "
    "and enterprise workflow orchestration. The future of AI lies not in isolated "
    "model calls but in sophisticated multi-step workflows that combine the reasoning "
    "power of LLMs with structured decision-making and real-world tool use."
)

SAMPLE_INVALID = ""

SAMPLE_TOO_SHORT = "Hi."


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------

@app.command()
def process(
    text: str = typer.Argument(..., help="Document text to process"),
):
    """Process a single document through the pipeline."""
    console.print(Rule("📄 Document Processing Graph — Lab 4.1", style="bold blue"))
    display_graph_structure()

    console.print()
    console.print(Rule("🚀 Executing Pipeline", style="bold green"))

    compiled = build_graph()
    result = compiled.invoke({
        "raw_text": text,
        "document_id": "",
        "is_valid": False,
        "validation_errors": [],
        "is_oversized": False,
        "parts": [],
        "chunks": [],
        "embeddings": [],
        "processing_log": [],
        "total_processing_time": 0.0,
    })

    display_result(result)


@app.command()
def process_file(
    filepath: str = typer.Argument(..., help="Path to a text file to process"),
):
    """Process a document from a file."""
    path = Path(filepath)
    if not path.exists():
        console.print(f"[red]❌ File not found: {filepath}[/]")
        raise typer.Exit(1)

    text = path.read_text(encoding="utf-8", errors="replace")
    console.print(f"[cyan]📂 Loaded file: {path.name} ({len(text):,} chars)[/]")
    process(text)


@app.command()
def graph():
    """Display the graph structure without processing."""
    console.print(Rule("📐 Document Processing Graph — Lab 4.1", style="bold blue"))
    display_graph_structure()


@app.command()
def demo():
    """Run demonstration with all sample document types."""
    console.print(Panel(
        "[bold cyan]Lab 4.1 — Document Processing Graph[/]\n"
        "[dim]LangGraph workflow with conditional routing for document processing[/]\n\n"
        "This demo shows how the graph handles:\n"
        "  1️⃣  A [green]normal-sized[/] document (direct chunking)\n"
        "  2️⃣  An [yellow]oversized[/] document (split → then chunk)\n"
        "  3️⃣  An [red]empty/invalid[/] document (rejected)\n"
        "  4️⃣  A [red]too-short[/] document (rejected)",
        title="🔬 Demo Mode",
        border_style="blue",
        padding=(1, 2),
    ))

    # Show graph structure
    display_graph_structure()

    compiled = build_graph()

    test_cases = [
        ("Normal Document (within size limit)", SAMPLE_SHORT, "green"),
        ("Oversized Document (exceeds threshold)", SAMPLE_OVERSIZED, "yellow"),
        ("Empty Document (invalid)", SAMPLE_INVALID, "red"),
        ("Too-Short Document (invalid)", SAMPLE_TOO_SHORT, "red"),
    ]

    for label, text, color in test_cases:
        console.print()
        console.print(Rule(f"🧪 Test: {label}", style=f"bold {color}"))
        console.print(f"[dim]Input length: {len(text):,} chars[/]")

        result = compiled.invoke({
            "raw_text": text,
            "document_id": "",
            "is_valid": False,
            "validation_errors": [],
            "is_oversized": False,
            "parts": [],
            "chunks": [],
            "embeddings": [],
            "processing_log": [],
            "total_processing_time": 0.0,
        })

        display_result(result)

    # Summary
    console.print()
    console.print(Panel(
        "[bold green]✅ All test cases completed![/]\n\n"
        "Key takeaways:\n"
        "  • [cyan]Conditional edges[/] route documents based on size\n"
        "  • [cyan]Oversized docs[/] are split before chunking\n"
        "  • [cyan]Invalid docs[/] are caught early and rejected\n"
        "  • [cyan]State[/] accumulates processing logs across all nodes",
        title="📊 Demo Summary",
        border_style="green",
    ))


# ---------------------------------------------------------------------------
# Entry Point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app()
