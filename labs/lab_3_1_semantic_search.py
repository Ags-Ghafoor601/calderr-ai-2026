#!/usr/bin/env python3
"""
Lab 3.1 — Semantic Search CLI
==============================
Embeds 100 Wikipedia sentences using sentence-transformers.
Builds a CLI tool for semantic search with rich terminal output.
Compares results between all-MiniLM-L6-v2 and BAAI/bge-small-en-v1.5.
Includes 2D PCA visualization of embedding spaces.

Usage:
    python lab_3_1_semantic_search.py search "your query" --model minilm --top-k 5
    python lab_3_1_semantic_search.py compare "your query"
    python lab_3_1_semantic_search.py visualize --model minilm
    python lab_3_1_semantic_search.py demo
"""

import io
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
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich import box

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
MODELS = {
    "minilm": "all-MiniLM-L6-v2",
    "bge": "BAAI/bge-small-en-v1.5",
}

console = Console()
app = typer.Typer(
    name="semantic-search",
    help="🔍 Semantic Search CLI — Lab 3.1",
    add_completion=False,
)

# ---------------------------------------------------------------------------
# 100 Wikipedia sentences (diverse topics)
# ---------------------------------------------------------------------------
WIKIPEDIA_SENTENCES = [
    # Science & Physics
    "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
    "Albert Einstein published his theory of general relativity in 1915.",
    "Quantum mechanics describes nature at the smallest scales of energy levels.",
    "The Higgs boson was discovered at CERN in 2012 using the Large Hadron Collider.",
    "Dark matter makes up approximately 27 percent of the universe's mass-energy content.",
    "Entropy is a measure of the number of possible arrangements in a thermodynamic system.",
    "The double-slit experiment demonstrates wave-particle duality of light and matter.",
    "Superconductivity is the complete disappearance of electrical resistance in certain materials.",
    # Biology & Medicine
    "DNA carries the genetic instructions used in the growth and functioning of all living organisms.",
    "The human genome contains approximately 20,000 to 25,000 protein-coding genes.",
    "Photosynthesis converts carbon dioxide and water into glucose using sunlight energy.",
    "CRISPR-Cas9 is a powerful tool for editing genomes, allowing precise changes to DNA.",
    "Mitochondria are often called the powerhouses of the cell because they generate ATP.",
    "The human brain contains approximately 86 billion neurons connected by trillions of synapses.",
    "Antibiotics were first discovered when Alexander Fleming noticed mold killing bacteria in 1928.",
    "Evolution by natural selection was proposed independently by Charles Darwin and Alfred Wallace.",
    # Computer Science & Technology
    "The first programmable electronic computer, ENIAC, was completed in 1945.",
    "Machine learning is a subset of artificial intelligence that learns patterns from data.",
    "The internet was originally developed as ARPANET by the US Department of Defense.",
    "Python is one of the most popular programming languages for data science and AI.",
    "Cloud computing delivers computing services over the internet on a pay-as-you-go basis.",
    "Blockchain technology creates a decentralized and distributed digital ledger of transactions.",
    "Neural networks are computing systems inspired by biological neural networks in the brain.",
    "The Turing test evaluates a machine's ability to exhibit intelligent behavior equivalent to a human.",
    # History
    "The Roman Empire at its greatest extent covered approximately 5 million square kilometers.",
    "The Renaissance began in Italy in the 14th century and spread across Europe.",
    "The French Revolution of 1789 was a period of radical political and societal change in France.",
    "World War II lasted from 1939 to 1945 and involved most of the world's nations.",
    "The Berlin Wall fell on November 9, 1989, symbolizing the end of the Cold War.",
    "The printing press was invented by Johannes Gutenberg around 1440 in Mainz, Germany.",
    "The Industrial Revolution began in Britain in the late 18th century with textile manufacturing.",
    "Ancient Egypt's civilization lasted for over 3,000 years along the Nile River valley.",
    # Geography & Nature
    "Mount Everest is the highest mountain on Earth at 8,849 meters above sea level.",
    "The Amazon River carries more water than any other river system in the world.",
    "The Sahara Desert is the largest hot desert in the world covering 9.2 million square kilometers.",
    "The Great Barrier Reef is the world's largest coral reef system visible from outer space.",
    "Antarctica is the coldest, driest, and windiest continent on Earth.",
    "The Pacific Ocean is the largest and deepest ocean covering more than 165 million square kilometers.",
    "The Amazon rainforest produces approximately 20 percent of the world's oxygen.",
    "The Mariana Trench is the deepest part of the ocean at approximately 11,034 meters deep.",
    # Mathematics
    "Pi is an irrational number approximately equal to 3.14159265358979.",
    "The Pythagorean theorem states that in a right triangle, a squared plus b squared equals c squared.",
    "Euler's identity connects five fundamental mathematical constants: e, i, pi, 1, and 0.",
    "The Fibonacci sequence appears frequently in nature, from flower petals to spiral galaxies.",
    "Calculus was independently developed by Isaac Newton and Gottfried Wilhelm Leibniz.",
    "Prime numbers are natural numbers greater than 1 that have no positive divisors other than 1 and themselves.",
    "The concept of zero as a number was developed in India around the 5th century.",
    "Game theory is the study of mathematical models of strategic interactions between rational agents.",
    # Arts & Culture
    "Leonardo da Vinci painted the Mona Lisa between 1503 and 1519 in Florence, Italy.",
    "Shakespeare wrote approximately 39 plays, 154 sonnets, and several longer poems.",
    "The Sistine Chapel ceiling was painted by Michelangelo between 1508 and 1512.",
    "Jazz music originated in the African-American communities of New Orleans in the early 20th century.",
    "The Great Wall of China was built over many centuries to protect against nomadic invasions.",
    "Beethoven composed his Ninth Symphony while completely deaf.",
    "The Louvre Museum in Paris is the world's most visited art museum with over 10 million visitors annually.",
    "Ancient Greek theater gave rise to the genres of tragedy and comedy.",
    # Economics & Politics
    "Gross domestic product measures the total value of goods and services produced in a country.",
    "The United Nations was established in 1945 to promote international cooperation and peace.",
    "Supply and demand is a fundamental economic model of price determination in a market.",
    "The European Union is a political and economic union of 27 member states in Europe.",
    "Inflation is the rate at which the general level of prices for goods and services rises.",
    "Democracy is a system of government where citizens exercise power by voting.",
    "The World Bank provides financial and technical assistance to developing countries.",
    "Free trade agreements reduce barriers to international trade between participating nations.",
    # Space & Astronomy
    "The Milky Way galaxy contains an estimated 100 to 400 billion stars.",
    "Mars is called the Red Planet because of iron oxide on its surface giving it a reddish appearance.",
    "A black hole is a region of spacetime where gravity is so strong that nothing can escape.",
    "The International Space Station orbits Earth approximately every 90 minutes.",
    "The nearest star to our Sun is Proxima Centauri at about 4.24 light-years away.",
    "Jupiter is the largest planet in our solar system with a mass greater than all other planets combined.",
    "The cosmic microwave background radiation is the thermal radiation left over from the Big Bang.",
    "Saturn's rings are primarily composed of ice particles and rocky debris.",
    # Philosophy & Psychology
    "Socrates is considered one of the founders of Western philosophy.",
    "Cognitive behavioral therapy is based on the idea that thoughts influence feelings and behaviors.",
    "Existentialism emphasizes individual existence, freedom, and the search for meaning.",
    "The placebo effect occurs when patients improve simply because they believe they are receiving treatment.",
    "Maslow's hierarchy of needs arranges human needs from physiological to self-actualization.",
    "Descartes's famous statement 'I think therefore I am' is a foundational element of Western philosophy.",
    "Pavlov's experiments with dogs demonstrated the concept of classical conditioning.",
    "The Trolley Problem is a thought experiment in ethics about sacrificing one person to save many.",
    # Environment & Climate
    "The greenhouse effect traps heat in Earth's atmosphere, maintaining habitable temperatures.",
    "Deforestation contributes to approximately 10 percent of global greenhouse gas emissions.",
    "Renewable energy sources include solar, wind, hydroelectric, and geothermal power.",
    "The ozone layer protects Earth from harmful ultraviolet radiation from the Sun.",
    "Ocean acidification occurs when carbon dioxide is absorbed by seawater, lowering its pH.",
    "Global average temperatures have risen approximately 1.1 degrees Celsius since the pre-industrial era.",
    "Coral bleaching occurs when ocean temperatures rise and corals expel their symbiotic algae.",
    "Wind energy is one of the fastest-growing sources of electricity generation worldwide.",
    # Sports
    "The modern Olympic Games were first held in Athens, Greece, in 1896.",
    "Football (soccer) is the most popular sport in the world with over 4 billion fans.",
    "The marathon race distance of 42.195 kilometers was standardized at the 1908 London Olympics.",
    "Cricket originated in England and is now popular across South Asia, Australia, and the Caribbean.",
    # Food & Agriculture
    "Rice is a staple food for more than half of the world's population.",
    "The Green Revolution of the 1960s dramatically increased agricultural production worldwide.",
    "Fermentation is a metabolic process that produces chemical changes in organic substrates.",
    "Coffee is the second most traded commodity in the world after crude oil.",
    # Language & Linguistics
    "Mandarin Chinese is the most spoken language in the world by number of native speakers.",
    "The Rosetta Stone was key to deciphering Egyptian hieroglyphics in the 19th century.",
    "There are approximately 7,000 languages spoken around the world today.",
    "Sign languages are complete natural languages with their own grammar and syntax.",
]

assert len(WIKIPEDIA_SENTENCES) == 100, f"Expected 100 sentences, got {len(WIKIPEDIA_SENTENCES)}"


# ---------------------------------------------------------------------------
# Embedding Engine
# ---------------------------------------------------------------------------
class SemanticSearchEngine:
    """Encapsulates embedding and similarity search for a given model."""

    def __init__(self, model_key: str = "minilm"):
        from sentence_transformers import SentenceTransformer

        self.model_key = model_key
        self.model_name = MODELS[model_key]
        self.sentences = WIKIPEDIA_SENTENCES

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold cyan]Loading model {task.fields[model]}..."),
            console=console,
        ) as progress:
            progress.add_task("load", model=self.model_name)
            self.model = SentenceTransformer(self.model_name)

        self.embeddings: Optional[np.ndarray] = None

    # ---- core ----
    def embed_sentences(self) -> np.ndarray:
        """Embed all 100 Wikipedia sentences and cache the result."""
        if self.embeddings is not None:
            return self.embeddings

        with Progress(
            SpinnerColumn(),
            TextColumn("[bold green]Embedding 100 sentences with {task.fields[model]}..."),
            console=console,
        ) as progress:
            progress.add_task("embed", model=self.model_name)
            start = time.perf_counter()
            self.embeddings = self.model.encode(
                self.sentences, show_progress_bar=False, normalize_embeddings=True
            )
            elapsed = time.perf_counter() - start

        console.print(
            f"  [dim]Embedded 100 sentences in [bold]{elapsed:.2f}s[/bold] "
            f"— dim={self.embeddings.shape[1]}[/dim]"
        )
        return self.embeddings

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Return top-k most similar sentences to the query."""
        embs = self.embed_sentences()
        query_emb = self.model.encode([query], normalize_embeddings=True)
        scores = np.dot(embs, query_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for rank, idx in enumerate(top_indices, 1):
            results.append(
                {
                    "rank": rank,
                    "score": float(scores[idx]),
                    "sentence": self.sentences[idx],
                    "index": int(idx),
                }
            )
        return results

    def visualize_pca(self, output_path: str = "pca_embeddings.png", query: str | None = None):
        """Generate a 2D PCA scatter plot of the embeddings."""
        from sklearn.decomposition import PCA
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        embs = self.embed_sentences()
        pca = PCA(n_components=2, random_state=42)
        reduced = pca.fit_transform(embs)

        # Assign topic colours (every 8 sentences is a topic group)
        topic_names = [
            "Physics", "Biology", "CompSci", "History",
            "Geography", "Math", "Arts", "Economics",
            "Space", "Philosophy", "Environment", "Sports/Food/Lang",
        ]
        n_per_group = 8
        colours = plt.cm.Set3(np.linspace(0, 1, len(topic_names)))
        point_colours = []
        for i in range(len(self.sentences)):
            group = min(i // n_per_group, len(topic_names) - 1)
            point_colours.append(colours[group])

        fig, ax = plt.subplots(figsize=(14, 10))
        ax.set_facecolor("#0d1117")
        fig.patch.set_facecolor("#0d1117")

        # Plot each topic group
        for g_idx, g_name in enumerate(topic_names):
            start_i = g_idx * n_per_group
            end_i = min(start_i + n_per_group, len(self.sentences))
            mask = range(start_i, end_i)
            ax.scatter(
                reduced[mask, 0],
                reduced[mask, 1],
                c=[colours[g_idx]],
                label=g_name,
                s=60,
                alpha=0.85,
                edgecolors="white",
                linewidths=0.3,
            )

        # Plot query if provided
        if query:
            q_emb = self.model.encode([query], normalize_embeddings=True)
            q_reduced = pca.transform(q_emb)
            ax.scatter(
                q_reduced[0, 0], q_reduced[0, 1],
                c="red", marker="*", s=300, zorder=10, label=f"Query: {query[:30]}"
            )

        ax.set_title(
            f"PCA of Embeddings — {self.model_name}",
            color="white", fontsize=16, fontweight="bold"
        )
        ax.set_xlabel("PC1", color="white")
        ax.set_ylabel("PC2", color="white")
        ax.tick_params(colors="white")
        ax.legend(
            loc="upper left", fontsize=8, facecolor="#161b22",
            edgecolor="white", labelcolor="white", ncol=2,
        )
        for spine in ax.spines.values():
            spine.set_color("#30363d")

        plt.tight_layout()
        plt.savefig(output_path, dpi=150, facecolor=fig.get_facecolor())
        plt.close()
        console.print(f"  [green]✓[/green] PCA plot saved → [bold]{output_path}[/bold]")


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def display_results(results: list[dict], model_name: str, query: str):
    """Pretty-print search results in a rich table."""
    table = Table(
        title=f"🔍 Results for: \"{query}\"",
        box=box.ROUNDED,
        title_style="bold magenta",
        border_style="bright_blue",
        show_lines=True,
    )
    table.add_column("Rank", style="bold cyan", justify="center", width=5)
    table.add_column("Score", style="bold green", justify="center", width=8)
    table.add_column("Sentence", style="white", ratio=1)

    for r in results:
        score_color = "green" if r["score"] > 0.5 else "yellow" if r["score"] > 0.3 else "red"
        table.add_row(
            str(r["rank"]),
            f"[{score_color}]{r['score']:.4f}[/{score_color}]",
            r["sentence"],
        )

    console.print()
    console.print(Panel(f"Model: [bold cyan]{model_name}[/bold cyan]", style="dim"))
    console.print(table)


# ---------------------------------------------------------------------------
# CLI Commands
# ---------------------------------------------------------------------------
@app.command()
def search(
    query: str = typer.Argument(..., help="Search query text"),
    model: str = typer.Option("minilm", "--model", "-m", help="Model key: minilm or bge"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results to return"),
):
    """Search for semantically similar sentences using a single model."""
    if model not in MODELS:
        console.print(f"[red]Unknown model '{model}'. Choose from: {list(MODELS.keys())}[/red]")
        raise typer.Exit(1)

    engine = SemanticSearchEngine(model)
    results = engine.search(query, top_k=top_k)
    display_results(results, engine.model_name, query)


@app.command()
def compare(
    query: str = typer.Argument(..., help="Search query text"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Number of results"),
):
    """Compare search results between both embedding models side by side."""
    console.print(
        Panel(
            "[bold]🔬 Model Comparison[/bold]\n"
            f"Query: [italic]\"{query}\"[/italic]\n"
            f"Comparing: [cyan]{MODELS['minilm']}[/cyan] vs [yellow]{MODELS['bge']}[/yellow]",
            style="bright_blue",
            border_style="bright_blue",
        )
    )

    engines = {}
    for key in MODELS:
        engines[key] = SemanticSearchEngine(key)

    results = {}
    for key, engine in engines.items():
        results[key] = engine.search(query, top_k=top_k)

    # Side-by-side comparison table
    table = Table(
        title="📊 Side-by-Side Comparison",
        box=box.DOUBLE_EDGE,
        title_style="bold magenta",
        border_style="bright_blue",
        show_lines=True,
    )
    table.add_column("Rank", style="bold", justify="center", width=5)
    table.add_column(f"MiniLM (Score)", style="cyan", width=10)
    table.add_column(f"MiniLM Result", style="white", ratio=1)
    table.add_column(f"BGE (Score)", style="yellow", width=10)
    table.add_column(f"BGE Result", style="white", ratio=1)

    for i in range(top_k):
        m_r = results["minilm"][i] if i < len(results["minilm"]) else {"score": 0, "sentence": "-"}
        b_r = results["bge"][i] if i < len(results["bge"]) else {"score": 0, "sentence": "-"}
        table.add_row(
            str(i + 1),
            f"{m_r['score']:.4f}",
            m_r["sentence"][:80] + ("..." if len(m_r["sentence"]) > 80 else ""),
            f"{b_r['score']:.4f}",
            b_r["sentence"][:80] + ("..." if len(b_r["sentence"]) > 80 else ""),
        )

    console.print(table)

    # Overlap analysis
    minilm_set = {r["index"] for r in results["minilm"]}
    bge_set = {r["index"] for r in results["bge"]}
    overlap = minilm_set & bge_set
    console.print(
        f"\n  [dim]Overlap: [bold]{len(overlap)}/{top_k}[/bold] sentences appear in both result sets[/dim]"
    )


@app.command()
def visualize(
    model: str = typer.Option("minilm", "--model", "-m", help="Model key: minilm or bge"),
    query: str = typer.Option(None, "--query", "-q", help="Optional query to highlight"),
    output: str = typer.Option("pca_embeddings.png", "--output", "-o", help="Output image path"),
):
    """Generate a 2D PCA visualization of the embedding space."""
    if model not in MODELS:
        console.print(f"[red]Unknown model '{model}'. Choose from: {list(MODELS.keys())}[/red]")
        raise typer.Exit(1)

    engine = SemanticSearchEngine(model)
    engine.visualize_pca(output_path=output, query=query)


@app.command()
def demo():
    """Run a comprehensive demonstration with multiple queries and both models."""
    demo_queries = [
        "How does the human brain work?",
        "What is climate change and its effects?",
        "Tell me about ancient civilizations",
        "How do computers learn from data?",
        "What are the planets in our solar system?",
    ]

    console.print(
        Panel(
            "[bold]🚀 Semantic Search Demo[/bold]\n"
            "Running 5 queries across both models with comparison",
            style="bright_magenta",
        )
    )

    for key in MODELS:
        engine = SemanticSearchEngine(key)

        console.print(f"\n{'='*80}")
        console.print(f"[bold]Model: {engine.model_name}[/bold]")
        console.print(f"{'='*80}")

        for q in demo_queries:
            results = engine.search(q, top_k=3)
            display_results(results, engine.model_name, q)

        # Save PCA plot
        out_file = f"pca_{key}.png"
        engine.visualize_pca(output_path=os.path.join(
            os.path.dirname(__file__), out_file
        ))

    # Final comparison on one query
    console.print(f"\n{'='*80}")
    console.print("[bold magenta]🔬 Final Comparison: 'How does artificial intelligence work?'[/bold magenta]")
    console.print(f"{'='*80}")

    for key in MODELS:
        engine = SemanticSearchEngine(key)
        results = engine.search("How does artificial intelligence work?", top_k=5)
        display_results(results, engine.model_name, "How does artificial intelligence work?")

    console.print("\n[bold green]✅ Demo complete![/bold green]")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app()
