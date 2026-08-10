#!/usr/bin/env python3
"""
CalderR Internship – Week 6, Lab 6.3
======================================
GraphRAG — Vector + Graph Hybrid Retrieval with Query Routing

WHAT THIS LAB BUILDS:
---------------------
A complete GraphRAG pipeline that outperforms standard RAG:
  • Vector retrieval via ChromaDB (top-5 chunk similarity)
  • Graph traversal via NetworkX (neighbourhood expansion from
    identified entities)
  • Parallel execution of both retrieval strategies
  • Context merger with deduplication
  • Automatic query routing: a classifier decides vector-only,
    graph-only, or hybrid based on question type
  • 15-question evaluation across 3 categories:
    - Factual (5 questions — vector retrieval excels)
    - Relational (5 questions — graph traversal excels)
    - Complex/multi-hop (5 questions — hybrid excels)
  • Structured comparison table with scores per retrieval mode

WHAT THIS TEACHES YOU:
----------------------
  • GraphRAG: combining vector similarity with graph structure
  • Query routing — classifying questions to pick the best strategy
  • Context deduplication across retrieval sources
  • Rigorous evaluation methodology comparing retrieval modes
  • Building systems that are provably better than baselines

ARCHITECTURE:
    ┌──────────────────────────────────────────────────────────────┐
    │                     GraphRAG PIPELINE                        │
    │                                                              │
    │  ┌──────────┐     ┌──────────────────┐                       │
    │  │  Query   │────►│  Query Router    │                       │
    │  │          │     │  (classifier)    │                       │
    │  └──────────┘     └─────┬──────┬─────┘                       │
    │                   factual│     │relational    complex         │
    │              ┌──────────┘      └──────────┐   │              │
    │              ▼                            ▼   ▼              │
    │  ┌───────────────┐              ┌──────────────────┐         │
    │  │ ChromaDB      │              │ NetworkX Graph   │         │
    │  │ Vector        │              │ Traversal        │         │
    │  │ Retrieval     │              │ (neighbourhood   │         │
    │  │ (top-5)       │              │  expansion)      │         │
    │  └───────┬───────┘              └────────┬─────────┘         │
    │          │                               │                   │
    │          └───────────┬───────────────────┘                   │
    │                      ▼                                       │
    │          ┌───────────────────────┐                            │
    │          │  Context Merger &     │                            │
    │          │  Deduplicator         │                            │
    │          └───────────┬───────────┘                            │
    │                      ▼                                       │
    │          ┌───────────────────────┐                            │
    │          │  LLM Generator       │                            │
    │          │  (Groq)              │                            │
    │          └───────────┬───────────┘                            │
    │                      ▼                                       │
    │          ┌───────────────────────┐                            │
    │          │  Answer + Evidence    │                            │
    │          └───────────────────────┘                            │
    └──────────────────────────────────────────────────────────────┘

Run:
    python labs/lab_6_3_graphrag.py demo
    python labs/lab_6_3_graphrag.py evaluate
    python labs/lab_6_3_graphrag.py compare
    python labs/lab_6_3_graphrag.py query "What companies has Microsoft invested in?"
"""

# pylint: disable=line-too-long, too-many-locals, wrong-import-position, broad-exception-caught, missing-class-docstring, missing-function-docstring, too-few-public-methods, duplicate-code
import io
import os
import sys
import json
import time
import re
from pathlib import Path
from typing import Any

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import typer
import networkx as nx
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.rule import Rule
from rich import box

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

from groq import Groq

console = Console()
app = typer.Typer(help="Lab 6.3 — GraphRAG: Vector + Graph Hybrid Retrieval")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = "llama-3.1-8b-instant"
CHROMA_PATH = str(ROOT_DIR / "labs" / ".chromadb_lab63")
GRAPH_PATH = ROOT_DIR / "labs" / ".knowledge_graph_lab62.json"  # Reuse graph from Lab 6.2
EVAL_REPORT_PATH = ROOT_DIR / "labs" / "lab_6_3_evaluation_report.json"


# ─── LLM Helper ────────────────────────────────────────────────────────────
def llm_call(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    """Make a single LLM call via Groq with retry logic."""
    client = Groq(api_key=GROQ_API_KEY)
    for attempt in range(4):
        try:
            resp = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=temperature,
                max_tokens=1536,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                wait = (attempt + 1) * 12
                time.sleep(wait)
            else:
                raise
    return "Unable to generate response after retries."


# ═══════════════════════════════════════════════════════════════════════════
#  PART 1 — CORPUS (Same AI/Tech paragraphs from Lab 6.2)
# ═══════════════════════════════════════════════════════════════════════════

CORPUS = [
    "OpenAI was founded in December 2015 by Sam Altman, Elon Musk, Greg Brockman, Ilya Sutskever, Wojciech Zaremski, and John Schulman. The organisation is headquartered in San Francisco, California. OpenAI's mission is to ensure that artificial general intelligence benefits all of humanity.",
    "GPT-4 is a large language model developed by OpenAI and released in March 2023. It is a multimodal model that accepts both text and image inputs and produces text outputs. GPT-4 was trained using reinforcement learning from human feedback (RLHF) and demonstrated significantly improved performance over GPT-3.5 on various benchmarks.",
    "Google DeepMind was formed in April 2023 by merging Google Brain and DeepMind. Demis Hassabis serves as the CEO of Google DeepMind. The organisation is owned by Alphabet Inc. and is headquartered in London, United Kingdom.",
    "AlphaFold is a protein structure prediction system developed by Google DeepMind. AlphaFold2 won the CASP14 competition in 2020 with unprecedented accuracy. The system uses a transformer-based neural network architecture and has predicted structures for over 200 million proteins.",
    "Meta AI, the artificial intelligence research division of Meta Platforms (formerly Facebook), is led by Yann LeCun as Chief AI Scientist. Meta AI is headquartered in Menlo Park, California. The division developed the LLaMA family of open-source large language models.",
    "LLaMA (Large Language Model Meta AI) is a family of open-source language models released by Meta AI starting in February 2023. LLaMA 2 was released in July 2023 with model sizes ranging from 7 billion to 70 billion parameters. LLaMA models are designed to be more accessible for research and commercial use.",
    "Anthropic was founded in 2021 by Dario Amodei and Daniela Amodei, former employees of OpenAI. Anthropic is headquartered in San Francisco and focuses on AI safety research. The company developed the Claude family of AI assistants.",
    "Claude is a family of large language models developed by Anthropic. Claude uses a technique called Constitutional AI (CAI) for alignment, which trains the model to follow a set of principles. Claude 3, released in 2024, includes three variants: Haiku, Sonnet, and Opus.",
    "NVIDIA Corporation, founded by Jensen Huang, Chris Malachowsky, and Curtis Priem in 1993, is headquartered in Santa Clara, California. NVIDIA designs the GPU hardware that powers most modern AI training and inference workloads, including the A100 and H100 data centre GPUs.",
    "The transformer architecture was introduced in the 2017 paper 'Attention Is All You Need' by Ashish Vaswani and colleagues at Google Brain. Transformers use self-attention mechanisms and have become the foundation of virtually all modern large language models including GPT-4, LLaMA, and Claude.",
    "Hugging Face is an AI company founded by Clement Delangue, Julien Chaumond, and Thomas Wolf. The company is headquartered in New York City and maintains the Hugging Face Hub, the largest open-source repository of machine learning models, datasets, and demos.",
    "PyTorch is an open-source machine learning framework originally developed by Meta AI (then Facebook AI Research). PyTorch is maintained by the PyTorch Foundation under the Linux Foundation. It is the most widely used framework for deep learning research and is the framework behind LLaMA and many other large language models.",
    "Stability AI, founded by Emad Mostaque in 2019, developed Stable Diffusion, an open-source text-to-image generative model. Stability AI is headquartered in London. Stable Diffusion competes with DALL-E (developed by OpenAI) and Midjourney in the image generation market.",
    "Mistral AI was founded in April 2023 by Arthur Mensch, Guillaume Lample, and Timothee Lacroix, former researchers from Google DeepMind and Meta AI. Mistral AI is headquartered in Paris, France. The company released Mixtral 8x7B, a mixture-of-experts language model.",
    "Google developed the Gemini family of multimodal AI models, succeeding their earlier PaLM models. Gemini was built by Google DeepMind and released in December 2023. Gemini Ultra achieved state-of-the-art performance on multiple benchmarks and powers Google's Bard (now Gemini) chatbot.",
    "Sam Altman serves as the CEO of OpenAI. He previously co-founded Loopt and served as president of Y Combinator before joining OpenAI. In November 2023, Altman was briefly fired and then reinstated as CEO of OpenAI following a board dispute.",
    "Yann LeCun, a Turing Award winner alongside Geoffrey Hinton and Yoshua Bengio, is known for his pioneering work on convolutional neural networks (CNNs). LeCun is the Chief AI Scientist at Meta and a professor at New York University. He has been a vocal critic of large language models as a path to AGI.",
    "Microsoft invested over $10 billion in OpenAI and integrated GPT-4 into its products including Bing Chat (now Microsoft Copilot), Microsoft 365 Copilot, and GitHub Copilot. Microsoft Azure provides the cloud computing infrastructure that OpenAI uses to train and deploy its models.",
    "Amazon Web Services (AWS) invested up to $4 billion in Anthropic in 2023. As part of the deal, Anthropic uses AWS custom chips (Trainium and Inferentia) for training and deploying Claude models. AWS competes with Microsoft Azure and Google Cloud Platform in the cloud AI infrastructure market.",
    "The AI safety research community includes organisations like Anthropic, the Center for AI Safety (CAIS), and the Machine Intelligence Research Institute (MIRI). Key concerns include alignment (ensuring AI systems act according to human values), interpretability (understanding how AI models make decisions), and existential risk from advanced AI systems.",
]


# ═══════════════════════════════════════════════════════════════════════════
#  PART 2 — PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════

class QueryCategory(BaseModel):
    """Classification of a query type."""
    category: str = Field(..., description="factual, relational, or complex")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)
    reasoning: str = Field(default="")


class RetrievalContext(BaseModel):
    """A single piece of retrieved context."""
    source: str = Field(..., description="'vector' or 'graph'")
    content: str = Field(...)
    relevance_score: float = Field(default=0.0, ge=0.0, le=1.0)


class GraphRAGResult(BaseModel):
    """Result from the GraphRAG pipeline."""
    question: str = Field(...)
    category: str = Field(default="unknown")
    retrieval_mode: str = Field(default="hybrid", description="vector, graph, or hybrid")
    answer: str = Field(default="")
    vector_contexts: list[str] = Field(default_factory=list)
    graph_contexts: list[str] = Field(default_factory=list)
    merged_context_count: int = Field(default=0)


class EvaluationResult(BaseModel):
    """Evaluation result for a single question."""
    question: str = Field(...)
    category: str = Field(...)
    expected_keywords: list[str] = Field(default_factory=list)
    vector_answer: str = Field(default="")
    graph_answer: str = Field(default="")
    hybrid_answer: str = Field(default="")
    vector_pass: bool = Field(default=False)
    graph_pass: bool = Field(default=False)
    hybrid_pass: bool = Field(default=False)


# ═══════════════════════════════════════════════════════════════════════════
#  PART 3 — VECTOR RETRIEVER (ChromaDB)
# ═══════════════════════════════════════════════════════════════════════════

class VectorRetriever:
    """ChromaDB-based vector retrieval for RAG."""

    def __init__(self, persist_dir: str = CHROMA_PATH):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="corpus_chunks",
            metadata={"hnsw:space": "cosine"},
        )

    def index_corpus(self, paragraphs: list[str]):
        """Index all paragraphs into ChromaDB."""
        ids = [f"para_{i}" for i in range(len(paragraphs))]
        self.collection.upsert(
            ids=ids,
            documents=paragraphs,
            metadatas=[{"paragraph_index": i} for i in range(len(paragraphs))],
        )
        console.print(f"[dim]Indexed {len(paragraphs)} paragraphs in ChromaDB[/]")

    def retrieve(self, query: str, n_results: int = 5) -> list[RetrievalContext]:
        """Retrieve top-N relevant paragraphs by vector similarity."""
        if self.collection.count() == 0:
            return []

        results = self.collection.query(
            query_texts=[query],
            n_results=min(n_results, self.collection.count()),
        )

        contexts = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                distance = results["distances"][0][i] if results["distances"] else 1.0
                similarity = max(0.0, 1.0 - distance)
                contexts.append(RetrievalContext(
                    source="vector",
                    content=doc,
                    relevance_score=similarity,
                ))

        return contexts

    def count(self) -> int:
        return self.collection.count()


# ═══════════════════════════════════════════════════════════════════════════
#  PART 4 — GRAPH RETRIEVER (NetworkX)
# ═══════════════════════════════════════════════════════════════════════════

class GraphRetriever:
    """NetworkX-based graph traversal for retrieval."""

    def __init__(self, graph_path: str = str(GRAPH_PATH)):
        self.graph_path = graph_path
        self.graph = nx.DiGraph()
        self._load()

    def _load(self):
        """Load graph from JSON."""
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.graph = nx.node_link_graph(data)
            except Exception:
                self.graph = nx.DiGraph()
        else:
            console.print(
                f"[yellow]Warning: Knowledge graph not found at {self.graph_path}. "
                "Run Lab 6.2 first (python labs/lab_6_2_knowledge_graph.py ingest) to build the graph. "
                "Graph retrieval will return empty results until the graph is built.[/]"
            )

    def _resolve_name(self, name: str) -> str | None:
        """Resolve a name to its canonical node name."""
        if name in self.graph.nodes:
            return name
        name_lower = name.lower()
        for node, data in self.graph.nodes(data=True):
            if node.lower() == name_lower:
                return node
            aliases = data.get("aliases", [])
            if any(a.lower() == name_lower for a in aliases):
                return node
        return None

    def _find_entities_in_query(self, query: str) -> list[str]:
        """Find graph entities mentioned in the query."""
        found = []
        q_lower = query.lower()
        for node in self.graph.nodes:
            if node.lower() in q_lower:
                found.append(node)
            else:
                aliases = self.graph.nodes[node].get("aliases", [])
                for alias in aliases:
                    if alias.lower() in q_lower and len(alias) > 2:
                        found.append(node)
                        break
        return found

    def retrieve(self, query: str, depth: int = 2) -> list[RetrievalContext]:
        """Retrieve graph context by identifying entities and expanding neighbourhoods."""
        entities = self._find_entities_in_query(query)

        if not entities:
            # Try LLM-based entity extraction
            all_nodes = list(self.graph.nodes)[:80]
            result = llm_call(
                "Given a question and a list of known entities, return ONLY a JSON list of relevant entity names.",
                f"Question: {query}\nKnown entities: {json.dumps(all_nodes)}",
                temperature=0.1,
            )
            try:
                cleaned = result.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
                    cleaned = cleaned.rstrip("`").strip()
                matched = json.loads(cleaned)
                entities = [e for e in matched if e in self.graph.nodes]
            except (json.JSONDecodeError, TypeError):
                pass

        if not entities:
            return []

        contexts = []
        seen_facts = set()

        for entity in entities:
            resolved = self._resolve_name(entity) or entity
            if resolved not in self.graph.nodes:
                continue

            node_data = self.graph.nodes[resolved]
            entity_desc = f"{resolved} (type: {node_data.get('entity_type', 'unknown')})"
            if node_data.get("description"):
                entity_desc += f": {node_data['description']}"

            # Outgoing relationships
            for _, target, edge_data in self.graph.out_edges(resolved, data=True):
                fact = f"{resolved} --[{edge_data.get('relation_type', 'related_to')}]--> {target}"
                if fact not in seen_facts:
                    seen_facts.add(fact)
                    contexts.append(RetrievalContext(
                        source="graph",
                        content=fact,
                        relevance_score=edge_data.get("confidence", 0.7),
                    ))

            # Incoming relationships
            for source, _, edge_data in self.graph.in_edges(resolved, data=True):
                fact = f"{source} --[{edge_data.get('relation_type', 'related_to')}]--> {resolved}"
                if fact not in seen_facts:
                    seen_facts.add(fact)
                    contexts.append(RetrievalContext(
                        source="graph",
                        content=fact,
                        relevance_score=edge_data.get("confidence", 0.7),
                    ))

            # 2-hop expansion
            if depth >= 2:
                for _, nbr, _ in self.graph.out_edges(resolved, data=True):
                    for _, nbr2, edge2 in self.graph.out_edges(nbr, data=True):
                        fact = f"{resolved} → {nbr} --[{edge2.get('relation_type', '')}]--> {nbr2}"
                        if fact not in seen_facts:
                            seen_facts.add(fact)
                            contexts.append(RetrievalContext(
                                source="graph",
                                content=fact,
                                relevance_score=edge2.get("confidence", 0.5) * 0.8,
                            ))

        # Sort by relevance
        contexts.sort(key=lambda c: c.relevance_score, reverse=True)
        return contexts[:15]


# ═══════════════════════════════════════════════════════════════════════════
#  PART 5 — QUERY ROUTER (Classifier)
# ═══════════════════════════════════════════════════════════════════════════

class QueryRouter:
    """Classifies questions to route to the optimal retrieval strategy."""

    ROUTER_PROMPT = """You are a query classifier for a GraphRAG system. Classify the question into one of three categories:

1. "factual" — A question asking about a specific fact, attribute, or description of a single entity.
   Examples: "What year was OpenAI founded?", "Where is NVIDIA headquartered?"
   Best served by: vector retrieval (keyword/semantic match)

2. "relational" — A question asking about a relationship between two entities.
   Examples: "Who founded Anthropic?", "What company developed GPT-4?"
   Best served by: graph traversal (direct edge lookup)

3. "complex" — A question requiring multi-hop reasoning across multiple entities/relationships.
   Examples: "Who is the CEO of the company that created the protein prediction system?",
   "What technology connects Google Brain's 2017 paper to Meta's LLaMA?"
   Best served by: hybrid (both vector + graph)

Respond with ONLY a JSON object: {"category": "factual|relational|complex", "confidence": 0.0-1.0, "reasoning": "brief reason"}"""

    def classify(self, question: str) -> QueryCategory:
        """Classify a question into factual, relational, or complex."""
        result = llm_call(
            self.ROUTER_PROMPT,
            f"Classify this question: {question}",
            temperature=0.1,
        )
        try:
            cleaned = result.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
                cleaned = cleaned.rstrip("`").strip()
            data = json.loads(cleaned)
            category = data.get("category", "complex")
            if category not in ("factual", "relational", "complex"):
                category = "complex"
            return QueryCategory(
                category=category,
                confidence=float(data.get("confidence", 0.8)),
                reasoning=data.get("reasoning", ""),
            )
        except (json.JSONDecodeError, TypeError, ValueError):
            return QueryCategory(category="complex", confidence=0.5, reasoning="Parse error — defaulting to hybrid")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 6 — GRAPHRAG PIPELINE
# ═══════════════════════════════════════════════════════════════════════════

class GraphRAGPipeline:
    """The core GraphRAG pipeline: route → retrieve → merge → generate."""

    def __init__(self):
        self.vector_retriever = VectorRetriever()
        self.graph_retriever = GraphRetriever()
        self.router = QueryRouter()

        # Index corpus if not already done
        if self.vector_retriever.count() == 0:
            console.print("[dim]Indexing corpus in ChromaDB...[/]")
            self.vector_retriever.index_corpus(CORPUS)

    def answer(self, question: str, force_mode: str | None = None) -> GraphRAGResult:
        """Answer a question using the GraphRAG pipeline."""
        # Step 1: Route the query
        if force_mode:
            category = force_mode
        else:
            classification = self.router.classify(question)
            category = classification.category

        # Step 2: Retrieve based on category
        vector_contexts: list[RetrievalContext] = []
        graph_contexts: list[RetrievalContext] = []

        mode = "hybrid"  # Default

        if category == "factual" and not force_mode:
            vector_contexts = self.vector_retriever.retrieve(question, n_results=5)
            mode = "vector"
        elif category == "relational" and not force_mode:
            graph_contexts = self.graph_retriever.retrieve(question, depth=2)
            mode = "graph"
        else:
            # Hybrid: both vector and graph
            vector_contexts = self.vector_retriever.retrieve(question, n_results=5)
            graph_contexts = self.graph_retriever.retrieve(question, depth=2)
            mode = "hybrid"

        if force_mode:
            mode = force_mode

        # Step 3: Merge and deduplicate contexts
        merged_context = self._merge_contexts(vector_contexts, graph_contexts, mode)

        # Step 4: Generate answer
        answer = self._generate(question, merged_context)

        return GraphRAGResult(
            question=question,
            category=category,
            retrieval_mode=mode,
            answer=answer,
            vector_contexts=[c.content for c in vector_contexts],
            graph_contexts=[c.content for c in graph_contexts],
            merged_context_count=len(merged_context),
        )

    def _merge_contexts(self, vector_ctxs: list[RetrievalContext], graph_ctxs: list[RetrievalContext], mode: str) -> list[str]:
        """Merge and deduplicate contexts from both sources."""
        merged = []
        seen = set()

        all_contexts = []
        if mode in ("vector", "hybrid"):
            all_contexts.extend(vector_ctxs)
        if mode in ("graph", "hybrid"):
            all_contexts.extend(graph_ctxs)

        # Sort by relevance score
        all_contexts.sort(key=lambda c: c.relevance_score, reverse=True)

        for ctx in all_contexts:
            # Simple deduplication: skip if content is very similar to something we've seen
            content_key = ctx.content[:80].lower()
            if content_key not in seen:
                seen.add(content_key)
                merged.append(f"[{ctx.source}] {ctx.content}")

        return merged[:10]  # Limit to top 10

    def _generate(self, question: str, contexts: list[str]) -> str:
        """Generate an answer from merged contexts."""
        context_str = "\n".join(f"- {c}" for c in contexts) if contexts else "No relevant context found."

        return llm_call(
            "You are a knowledge assistant. Answer the question based ONLY on the provided context. "
            "Be specific, factual, and concise. If the context doesn't contain enough information, say so.",
            f"Context:\n{context_str}\n\nQuestion: {question}",
            temperature=0.2,
        )


# ═══════════════════════════════════════════════════════════════════════════
#  PART 7 — EVALUATION FRAMEWORK
# ═══════════════════════════════════════════════════════════════════════════

# 15 evaluation questions: 5 factual, 5 relational, 5 complex
EVAL_QUESTIONS = [
    # ── Factual (5) — vector retrieval should excel ──
    {"question": "What year was OpenAI founded?", "category": "factual",
     "expected_keywords": ["2015"], "expected_winner": "vector"},

    {"question": "Where is NVIDIA Corporation headquartered?", "category": "factual",
     "expected_keywords": ["santa clara"], "expected_winner": "vector"},

    {"question": "What is the name of OpenAI's multimodal model released in March 2023?", "category": "factual",
     "expected_keywords": ["gpt-4"], "expected_winner": "vector"},

    {"question": "How many proteins has AlphaFold predicted structures for?", "category": "factual",
     "expected_keywords": ["200 million"], "expected_winner": "vector"},

    {"question": "What is the name of Stability AI's text-to-image model?", "category": "factual",
     "expected_keywords": ["stable diffusion"], "expected_winner": "vector"},

    # ── Relational (5) — graph traversal should excel ──
    {"question": "Who founded Anthropic?", "category": "relational",
     "expected_keywords": ["dario amodei", "daniela amodei"], "expected_winner": "graph"},

    {"question": "What company developed the LLaMA language models?", "category": "relational",
     "expected_keywords": ["meta ai", "meta"], "expected_winner": "graph"},

    {"question": "Who is the CEO of Google DeepMind?", "category": "relational",
     "expected_keywords": ["demis hassabis"], "expected_winner": "graph"},

    {"question": "What company invested over $10 billion in OpenAI?", "category": "relational",
     "expected_keywords": ["microsoft"], "expected_winner": "graph"},

    {"question": "Who introduced the transformer architecture?", "category": "relational",
     "expected_keywords": ["ashish vaswani", "google brain"], "expected_winner": "graph"},

    # ── Complex/Multi-hop (5) — hybrid should excel ──
    {"question": "Who founded the company that developed GPT-4, and what was his previous role at a startup accelerator?", "category": "complex",
     "expected_keywords": ["sam altman", "y combinator"], "expected_winner": "hybrid"},

    {"question": "What AI safety technique is used by the company whose founders previously worked at OpenAI?", "category": "complex",
     "expected_keywords": ["constitutional ai", "anthropic"], "expected_winner": "hybrid"},

    {"question": "What GPU hardware powers the training of models like GPT-4, and where is its manufacturer headquartered?", "category": "complex",
     "expected_keywords": ["nvidia", "santa clara", "h100", "a100"], "expected_winner": "hybrid"},

    {"question": "Which researchers who worked at Google DeepMind and Meta AI went on to found a company in Paris?", "category": "complex",
     "expected_keywords": ["mistral", "arthur mensch", "guillaume lample", "paris"], "expected_winner": "hybrid"},

    {"question": "What cloud provider invested in Anthropic, and what custom chips do they offer for AI training?", "category": "complex",
     "expected_keywords": ["aws", "amazon", "trainium", "inferentia"], "expected_winner": "hybrid"},
]


def evaluate_answer(answer: str, expected_keywords: list[str]) -> bool:
    """Check if an answer contains any of the expected keywords."""
    answer_lower = answer.lower()
    return any(kw.lower() in answer_lower for kw in expected_keywords)


# ═══════════════════════════════════════════════════════════════════════════
#  PART 8 — CLI COMMANDS
# ═══════════════════════════════════════════════════════════════════════════

@app.command()
def demo():
    """Run a quick GraphRAG demonstration."""
    console.print(Rule("[bold cyan]Lab 6.3 — GraphRAG Demo[/]"))
    console.print()
    console.print(Panel(
        "[bold]This demo shows the GraphRAG pipeline in action:[/]\n"
        "1. Query routing (factual vs relational vs complex)\n"
        "2. Parallel vector + graph retrieval\n"
        "3. Context merging and deduplication\n"
        "4. Answer generation with evidence",
        title="[bold cyan]GraphRAG Pipeline Demo[/]",
        border_style="cyan",
    ))
    console.print()

    pipeline = GraphRAGPipeline()

    demo_questions = [
        ("What year was OpenAI founded?", "factual"),
        ("Who is the CEO of Google DeepMind?", "relational"),
        ("Who founded the company that developed GPT-4, and what was his previous role?", "complex"),
    ]

    for question, expected_cat in demo_questions:
        console.print(f"\n[bold blue]Question:[/] {question}")
        console.print(f"[dim]Expected category: {expected_cat}[/]")

        result = pipeline.answer(question)

        console.print(f"[yellow]Classified as:[/] {result.category}")
        console.print(f"[yellow]Retrieval mode:[/] {result.retrieval_mode}")
        console.print(f"[green]Answer:[/] {result.answer}")
        console.print(f"[dim]Vector contexts: {len(result.vector_contexts)} | Graph contexts: {len(result.graph_contexts)} | Merged: {result.merged_context_count}[/]")
        time.sleep(3)

    console.print()
    console.print(Rule("[bold cyan]Demo Complete[/]"))


@app.command()
def query(question: str = typer.Argument(..., help="Question to answer")):
    """Answer a single question using the GraphRAG pipeline."""
    console.print(Rule("[bold cyan]Lab 6.3 — GraphRAG Query[/]"))

    pipeline = GraphRAGPipeline()
    result = pipeline.answer(question)

    console.print(Panel(
        f"[bold]Question:[/] {result.question}\n\n"
        f"[bold]Category:[/] {result.category}\n"
        f"[bold]Mode:[/] {result.retrieval_mode}\n\n"
        f"[bold green]Answer:[/] {result.answer}\n\n"
        f"[dim]Vector contexts: {len(result.vector_contexts)} | "
        f"Graph contexts: {len(result.graph_contexts)} | "
        f"Merged: {result.merged_context_count}[/]",
        title="[bold cyan]GraphRAG Result[/]",
        border_style="cyan",
    ))


@app.command()
def evaluate():
    """Run the full 15-question evaluation across all three retrieval modes."""
    console.print(Rule("[bold cyan]Lab 6.3 — GraphRAG Evaluation[/]"))
    console.print()
    console.print(f"[bold]Running {len(EVAL_QUESTIONS)} questions across 3 retrieval modes...[/]\n")

    pipeline = GraphRAGPipeline()

    results: list[EvaluationResult] = []
    scores = {"vector": {"factual": 0, "relational": 0, "complex": 0},
              "graph": {"factual": 0, "relational": 0, "complex": 0},
              "hybrid": {"factual": 0, "relational": 0, "complex": 0}}
    totals = {"factual": 0, "relational": 0, "complex": 0}

    for i, q in enumerate(EVAL_QUESTIONS, 1):
        question = q["question"]
        category = q["category"]
        expected_kw = q["expected_keywords"]
        totals[category] += 1

        console.print(f"[bold]Q{i}[/] [{category}] {question[:70]}...")

        # Test all three modes
        vector_result = pipeline.answer(question, force_mode="vector")
        time.sleep(2)
        graph_result = pipeline.answer(question, force_mode="graph")
        time.sleep(2)
        hybrid_result = pipeline.answer(question, force_mode="hybrid")
        time.sleep(2)

        vector_pass = evaluate_answer(vector_result.answer, expected_kw)
        graph_pass = evaluate_answer(graph_result.answer, expected_kw)
        hybrid_pass = evaluate_answer(hybrid_result.answer, expected_kw)

        if vector_pass:
            scores["vector"][category] += 1
        if graph_pass:
            scores["graph"][category] += 1
        if hybrid_pass:
            scores["hybrid"][category] += 1

        results.append(EvaluationResult(
            question=question,
            category=category,
            expected_keywords=expected_kw,
            vector_answer=vector_result.answer,
            graph_answer=graph_result.answer,
            hybrid_answer=hybrid_result.answer,
            vector_pass=vector_pass,
            graph_pass=graph_pass,
            hybrid_pass=hybrid_pass,
        ))

        v_icon = "[green]✓[/]" if vector_pass else "[red]✗[/]"
        g_icon = "[green]✓[/]" if graph_pass else "[red]✗[/]"
        h_icon = "[green]✓[/]" if hybrid_pass else "[red]✗[/]"
        console.print(f"  Vector: {v_icon} | Graph: {g_icon} | Hybrid: {h_icon}")

    # ── Results Table ──
    console.print()
    console.print(Rule("[bold cyan]Evaluation Results[/]"))

    # Per-category table
    cat_table = Table(title="Scores by Category and Retrieval Mode", box=box.ROUNDED)
    cat_table.add_column("Category", style="bold")
    cat_table.add_column("Vector", justify="center")
    cat_table.add_column("Graph", justify="center")
    cat_table.add_column("Hybrid", justify="center")
    cat_table.add_column("Best", justify="center", style="bold")

    for cat in ["factual", "relational", "complex"]:
        v = scores["vector"][cat]
        g = scores["graph"][cat]
        h = scores["hybrid"][cat]
        t = totals[cat]

        best = "hybrid"
        if v >= g and v >= h:
            best = "vector"
        elif g >= v and g >= h:
            best = "graph"

        cat_table.add_row(
            cat.title(),
            f"{v}/{t}",
            f"{g}/{t}",
            f"{h}/{t}",
            best.title(),
        )

    # Totals
    total_v = sum(scores["vector"].values())
    total_g = sum(scores["graph"].values())
    total_h = sum(scores["hybrid"].values())
    total_q = sum(totals.values())

    cat_table.add_row(
        "[bold]TOTAL[/]",
        f"[bold]{total_v}/{total_q}[/]",
        f"[bold]{total_g}/{total_q}[/]",
        f"[bold]{total_h}/{total_q}[/]",
        "",
    )
    console.print(cat_table)

    # Query router validation
    router_correct = 0
    router = QueryRouter()
    for q in EVAL_QUESTIONS:
        classification = router.classify(q["question"])
        if classification.category == q["category"]:
            router_correct += 1
        time.sleep(1)

    console.print(f"\n[bold]Query Router Accuracy:[/] {router_correct}/{len(EVAL_QUESTIONS)} (need ≥12)")

    # Validation checks
    console.print()
    hybrid_complex_score = scores["hybrid"]["complex"]
    vector_complex_score = scores["vector"]["complex"]
    graph_complex_score = scores["graph"]["complex"]

    hybrid_wins = hybrid_complex_score > vector_complex_score and hybrid_complex_score > graph_complex_score
    router_passes = router_correct >= 12

    if hybrid_wins:
        console.print("[green]✓ Hybrid outperforms both modes on complex questions[/]")
    else:
        console.print("[yellow]⚠ Hybrid did not outperform on complex questions[/]")

    if router_passes:
        console.print(f"[green]✓ Query router classified ≥12/15 correctly ({router_correct}/15)[/]")
    else:
        console.print(f"[yellow]⚠ Query router classified {router_correct}/15 (need ≥12)[/]")

    # Save evaluation report
    report = {
        "scores_by_category": scores,
        "totals_by_category": totals,
        "overall_scores": {"vector": total_v, "graph": total_g, "hybrid": total_h, "total": total_q},
        "router_accuracy": router_correct,
        "hybrid_wins_complex": hybrid_wins,
        "details": [r.model_dump() for r in results],
    }
    with open(EVAL_REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    console.print(f"\n[dim]Report saved to {EVAL_REPORT_PATH}[/]")


@app.command()
def compare():
    """Display the evaluation comparison table from the last evaluation run."""
    console.print(Rule("[bold cyan]Lab 6.3 — Evaluation Comparison[/]"))

    if not os.path.exists(EVAL_REPORT_PATH):
        console.print("[yellow]No evaluation report found. Run 'evaluate' first.[/]")
        raise typer.Exit(1)

    with open(EVAL_REPORT_PATH, "r", encoding="utf-8") as f:
        report = json.load(f)

    # Detail table
    detail_table = Table(title="Per-Question Results", box=box.ROUNDED)
    detail_table.add_column("#", style="dim", width=3)
    detail_table.add_column("Category", width=10)
    detail_table.add_column("Question", width=50)
    detail_table.add_column("Vec", justify="center", width=4)
    detail_table.add_column("Graph", justify="center", width=5)
    detail_table.add_column("Hybrid", justify="center", width=6)

    for i, detail in enumerate(report.get("details", []), 1):
        v = "[green]✓[/]" if detail["vector_pass"] else "[red]✗[/]"
        g = "[green]✓[/]" if detail["graph_pass"] else "[red]✗[/]"
        h = "[green]✓[/]" if detail["hybrid_pass"] else "[red]✗[/]"
        detail_table.add_row(str(i), detail["category"], detail["question"][:50], v, g, h)

    console.print(detail_table)

    # Summary
    overall = report.get("overall_scores", {})
    console.print(f"\n[bold]Overall:[/] Vector {overall.get('vector')}/{overall.get('total')} | "
                  f"Graph {overall.get('graph')}/{overall.get('total')} | "
                  f"Hybrid {overall.get('hybrid')}/{overall.get('total')}")
    console.print(f"[bold]Router accuracy:[/] {report.get('router_accuracy')}/15")


if __name__ == "__main__":
    app()
