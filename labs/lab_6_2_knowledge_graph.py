#!/usr/bin/env python3
"""
CalderR Internship – Week 6, Lab 6.2
======================================
Knowledge Graph Query Agent — Entity Extraction, Graph Construction, Multi-Hop Queries

WHAT THIS LAB BUILDS:
---------------------
A complete knowledge extraction and query pipeline:
  • Feeds 20 text paragraphs (AI/tech domain) to an LLM that extracts
    entities (people, companies, concepts, technologies) and relationships
    (founded_by, works_at, developed, competes_with, etc.)
  • Stores the graph in NetworkX with rich node/edge attributes
  • Persists the graph as JSON on disk for cross-session use
  • Query agent converts natural language questions into graph traversals
  • Answers multi-hop questions (crossing 2+ edges)
  • Generates interactive pyvis HTML graph visualisation
  • Validates against 5 multi-hop questions (≥4 must pass)

WHAT THIS TEACHES YOU:
----------------------
  • LLM-driven entity and relationship extraction
  • NetworkX graph construction with typed nodes and edges
  • Multi-hop graph traversal for question answering
  • Entity deduplication and resolution
  • pyvis interactive visualisation
  • Comparing graph-based QA vs keyword-based QA

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────────┐
    │              KNOWLEDGE GRAPH QUERY AGENT                    │
    │                                                             │
    │  ┌──────────────┐     ┌───────────────────┐                 │
    │  │ 20 Text      │     │  Entity-Relation  │                 │
    │  │ Paragraphs   │────►│  Extractor (LLM)  │                 │
    │  │ (AI/Tech)    │     │  ─────────────    │                 │
    │  └──────────────┘     │  Pydantic schemas │                 │
    │                       └────────┬──────────┘                 │
    │                                │                            │
    │                       ┌────────▼──────────┐                 │
    │                       │  Deduplication &   │                 │
    │                       │  Entity Resolution │                 │
    │                       └────────┬──────────┘                 │
    │                                │                            │
    │                       ┌────────▼──────────┐                 │
    │                       │   NetworkX Graph   │                 │
    │                       │  (persisted JSON)  │                 │
    │                       └────────┬──────────┘                 │
    │                                │                            │
    │  ┌──────────────┐     ┌────────▼──────────┐                 │
    │  │  User Query   │───►│  Query Agent      │                 │
    │  │  (natural     │    │  ─────────────    │                 │
    │  │   language)   │    │  NL → graph       │                 │
    │  └──────────────┘     │  traversal        │                 │
    │                       └────────┬──────────┘                 │
    │                                │                            │
    │                       ┌────────▼──────────┐                 │
    │                       │  Answer + Path    │                 │
    │                       │  (reasoning chain)│                 │
    │                       └───────────────────┘                 │
    └─────────────────────────────────────────────────────────────┘

Run:
    python labs/lab_6_2_knowledge_graph.py demo
    python labs/lab_6_2_knowledge_graph.py ingest
    python labs/lab_6_2_knowledge_graph.py query "Who founded the company that developed GPT-4?"
    python labs/lab_6_2_knowledge_graph.py visualise
    python labs/lab_6_2_knowledge_graph.py validate
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
from rich.tree import Tree
from rich.rule import Rule
from rich import box

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

from groq import Groq

console = Console()
app = typer.Typer(help="Lab 6.2 — Knowledge Graph Query Agent")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = "llama-3.3-70b-versatile"
GRAPH_PATH = ROOT_DIR / "labs" / ".knowledge_graph_lab62.json"
PYVIS_PATH = ROOT_DIR / "labs" / "knowledge_graph_lab62.html"


# ─── LLM Helper ────────────────────────────────────────────────────────────
def llm_call(system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
    """Make a single LLM call via Groq with retry logic for rate limits."""
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
                max_tokens=2048,
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
#  PART 1 — KNOWLEDGE CORPUS (20 PARAGRAPHS — AI/TECH DOMAIN)
# ═══════════════════════════════════════════════════════════════════════════

CORPUS = [
    # 1
    "OpenAI was founded in December 2015 by Sam Altman, Elon Musk, Greg Brockman, Ilya Sutskever, Wojciech Zaremski, and John Schulman. The organisation is headquartered in San Francisco, California. OpenAI's mission is to ensure that artificial general intelligence benefits all of humanity.",

    # 2
    "GPT-4 is a large language model developed by OpenAI and released in March 2023. It is a multimodal model that accepts both text and image inputs and produces text outputs. GPT-4 was trained using reinforcement learning from human feedback (RLHF) and demonstrated significantly improved performance over GPT-3.5 on various benchmarks.",

    # 3
    "Google DeepMind was formed in April 2023 by merging Google Brain and DeepMind. Demis Hassabis serves as the CEO of Google DeepMind. The organisation is owned by Alphabet Inc. and is headquartered in London, United Kingdom.",

    # 4
    "AlphaFold is a protein structure prediction system developed by Google DeepMind. AlphaFold2 won the CASP14 competition in 2020 with unprecedented accuracy. The system uses a transformer-based neural network architecture and has predicted structures for over 200 million proteins.",

    # 5
    "Meta AI, the artificial intelligence research division of Meta Platforms (formerly Facebook), is led by Yann LeCun as Chief AI Scientist. Meta AI is headquartered in Menlo Park, California. The division developed the LLaMA family of open-source large language models.",

    # 6
    "LLaMA (Large Language Model Meta AI) is a family of open-source language models released by Meta AI starting in February 2023. LLaMA 2 was released in July 2023 with model sizes ranging from 7 billion to 70 billion parameters. LLaMA models are designed to be more accessible for research and commercial use.",

    # 7
    "Anthropic was founded in 2021 by Dario Amodei and Daniela Amodei, former employees of OpenAI. Anthropic is headquartered in San Francisco and focuses on AI safety research. The company developed the Claude family of AI assistants.",

    # 8
    "Claude is a family of large language models developed by Anthropic. Claude uses a technique called Constitutional AI (CAI) for alignment, which trains the model to follow a set of principles. Claude 3, released in 2024, includes three variants: Haiku, Sonnet, and Opus.",

    # 9
    "NVIDIA Corporation, founded by Jensen Huang, Chris Malachowsky, and Curtis Priem in 1993, is headquartered in Santa Clara, California. NVIDIA designs the GPU hardware that powers most modern AI training and inference workloads, including the A100 and H100 data centre GPUs.",

    # 10
    "The transformer architecture was introduced in the 2017 paper 'Attention Is All You Need' by Ashish Vaswani and colleagues at Google Brain. Transformers use self-attention mechanisms and have become the foundation of virtually all modern large language models including GPT-4, LLaMA, and Claude.",

    # 11
    "Hugging Face is an AI company founded by Clement Delangue, Julien Chaumond, and Thomas Wolf. The company is headquartered in New York City and maintains the Hugging Face Hub, the largest open-source repository of machine learning models, datasets, and demos.",

    # 12
    "PyTorch is an open-source machine learning framework originally developed by Meta AI (then Facebook AI Research). PyTorch is maintained by the PyTorch Foundation under the Linux Foundation. It is the most widely used framework for deep learning research and is the framework behind LLaMA and many other large language models.",

    # 13
    "Stability AI, founded by Emad Mostaque in 2019, developed Stable Diffusion, an open-source text-to-image generative model. Stability AI is headquartered in London. Stable Diffusion competes with DALL-E (developed by OpenAI) and Midjourney in the image generation market.",

    # 14
    "Mistral AI was founded in April 2023 by Arthur Mensch, Guillaume Lample, and Timothee Lacroix, former researchers from Google DeepMind and Meta AI. Mistral AI is headquartered in Paris, France. The company released Mixtral 8x7B, a mixture-of-experts language model.",

    # 15
    "Google developed the Gemini family of multimodal AI models, succeeding their earlier PaLM models. Gemini was built by Google DeepMind and released in December 2023. Gemini Ultra achieved state-of-the-art performance on multiple benchmarks and powers Google's Bard (now Gemini) chatbot.",

    # 16
    "Sam Altman serves as the CEO of OpenAI. He previously co-founded Loopt and served as president of Y Combinator before joining OpenAI. In November 2023, Altman was briefly fired and then reinstated as CEO of OpenAI following a board dispute.",

    # 17
    "Yann LeCun, a Turing Award winner alongside Geoffrey Hinton and Yoshua Bengio, is known for his pioneering work on convolutional neural networks (CNNs). LeCun is the Chief AI Scientist at Meta and a professor at New York University. He has been a vocal critic of large language models as a path to AGI.",

    # 18
    "Microsoft invested over $10 billion in OpenAI and integrated GPT-4 into its products including Bing Chat (now Microsoft Copilot), Microsoft 365 Copilot, and GitHub Copilot. Microsoft Azure provides the cloud computing infrastructure that OpenAI uses to train and deploy its models.",

    # 19
    "Amazon Web Services (AWS) invested up to $4 billion in Anthropic in 2023. As part of the deal, Anthropic uses AWS custom chips (Trainium and Inferentia) for training and deploying Claude models. AWS competes with Microsoft Azure and Google Cloud Platform in the cloud AI infrastructure market.",

    # 20
    "The AI safety research community includes organisations like Anthropic, the Center for AI Safety (CAIS), and the Machine Intelligence Research Institute (MIRI). Key concerns include alignment (ensuring AI systems act according to human values), interpretability (understanding how AI models make decisions), and existential risk from advanced AI systems.",
]


# ═══════════════════════════════════════════════════════════════════════════
#  PART 2 — PYDANTIC SCHEMAS FOR EXTRACTION
# ═══════════════════════════════════════════════════════════════════════════

class Entity(BaseModel):
    """An extracted entity from text."""
    name: str = Field(..., description="Canonical name of the entity")
    entity_type: str = Field(..., description="Type: person, company, technology, concept, place, product")
    description: str = Field(default="", description="Brief description of the entity")
    aliases: list[str] = Field(default_factory=list, description="Alternative names or abbreviations")


class Relationship(BaseModel):
    """An extracted relationship between two entities."""
    source: str = Field(..., description="Source entity name")
    target: str = Field(..., description="Target entity name")
    relation_type: str = Field(..., description="Type of relationship")
    evidence: str = Field(default="", description="Text evidence supporting this relationship")
    confidence: float = Field(default=0.8, ge=0.0, le=1.0)


class ExtractionResult(BaseModel):
    """Complete extraction result from a text paragraph."""
    entities: list[Entity] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    source_paragraph_index: int = Field(default=0)


class GraphTraversalStep(BaseModel):
    """A single step in a graph traversal path."""
    node: str = Field(..., description="Current node name")
    edge_type: str = Field(default="", description="Relationship type to next node")
    next_node: str = Field(default="", description="Next node in the path")


class QueryResult(BaseModel):
    """Result of a knowledge graph query."""
    question: str = Field(...)
    answer: str = Field(...)
    traversal_path: list[GraphTraversalStep] = Field(default_factory=list)
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    hops: int = Field(default=0, description="Number of edges traversed")
    nodes_visited: list[str] = Field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════
#  PART 3 — ENTITY-RELATIONSHIP EXTRACTOR
# ═══════════════════════════════════════════════════════════════════════════

class EntityRelationExtractor:
    """Uses LLM to extract entities and relationships from text paragraphs."""

    EXTRACTION_SYSTEM_PROMPT = """You are an entity and relationship extraction engine.
Given a text paragraph, extract ALL entities and relationships.

Entity types: person, company, technology, concept, place, product
Relationship types: founded_by, ceo_of, works_at, developed_by, headquartered_in,
    competes_with, invested_in, acquired, part_of, based_on, succeeded_by,
    uses, powers, released_by, won, trained_with

Return a valid JSON object with this EXACT structure:
{
  "entities": [
    {"name": "Entity Name", "entity_type": "type", "description": "brief desc", "aliases": ["alias1"]}
  ],
  "relationships": [
    {"source": "Entity A", "target": "Entity B", "relation_type": "relation", "evidence": "text evidence", "confidence": 0.9}
  ]
}

Rules:
- Use canonical names (e.g., "OpenAI" not "openai")
- Every relationship must reference entities by their exact canonical name
- Include ALL entities and relationships, even if implicit
- Confidence: 0.9+ for explicitly stated, 0.6-0.8 for implied
- Return ONLY valid JSON, no markdown, no explanation"""

    def extract(self, text: str, paragraph_index: int = 0) -> ExtractionResult:
        """Extract entities and relationships from a text paragraph."""
        for attempt in range(3):
            result = llm_call(
                self.EXTRACTION_SYSTEM_PROMPT,
                f"Extract entities and relationships from this text:\n\n{text}",
                temperature=0.1 + (attempt * 0.2),
            )
    
            try:
                # Clean up common JSON issues
                cleaned = result.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
                    cleaned = cleaned.rstrip("`").strip()
    
                data = json.loads(cleaned)
                entities = [Entity(**e) for e in data.get("entities", [])]
                relationships = [Relationship(**r) for r in data.get("relationships", [])]
                return ExtractionResult(
                    entities=entities,
                    relationships=relationships,
                    source_paragraph_index=paragraph_index,
                )
            except (json.JSONDecodeError, Exception) as e:
                if attempt == 2:
                    console.print(f"[yellow]Warning: Extraction parse error for paragraph {paragraph_index} after 3 attempts: {e}[/]")
                    
        return ExtractionResult(source_paragraph_index=paragraph_index)


# ═══════════════════════════════════════════════════════════════════════════
#  PART 4 — KNOWLEDGE GRAPH (NetworkX + JSON persistence)
# ═══════════════════════════════════════════════════════════════════════════

class KnowledgeGraph:
    """NetworkX-based knowledge graph with JSON persistence."""

    def __init__(self, graph_path: str = str(GRAPH_PATH)):
        self.graph_path = graph_path
        self.graph = nx.DiGraph()
        self._load()

    def _load(self):
        """Load graph from JSON file if it exists."""
        if os.path.exists(self.graph_path):
            try:
                with open(self.graph_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.graph = nx.node_link_graph(data)
                console.print(f"[dim]Loaded graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges[/]")
            except Exception as e:
                console.print(f"[yellow]Warning: Could not load graph: {e}. Starting fresh.[/]")
                self.graph = nx.DiGraph()

    def save(self):
        """Save graph to JSON file."""
        data = nx.node_link_data(self.graph)
        with open(self.graph_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        console.print(f"[dim]Graph saved: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges[/]")

    def add_entity(self, entity: Entity):
        """Add or update an entity node in the graph."""
        canonical = self._resolve_name(entity.name)
        if canonical and canonical != entity.name:
            # Entity already exists with a different canonical name — merge
            node_data = self.graph.nodes[canonical]
            existing_aliases = set(node_data.get("aliases", []))
            existing_aliases.add(entity.name)
            existing_aliases.update(entity.aliases)
            node_data["aliases"] = list(existing_aliases)
            if entity.description and not node_data.get("description"):
                node_data["description"] = entity.description
        else:
            # New entity or exact match
            if entity.name in self.graph.nodes:
                # Update existing
                node_data = self.graph.nodes[entity.name]
                existing_aliases = set(node_data.get("aliases", []))
                existing_aliases.update(entity.aliases)
                node_data["aliases"] = list(existing_aliases)
                if entity.description:
                    node_data["description"] = entity.description
            else:
                self.graph.add_node(
                    entity.name,
                    entity_type=entity.entity_type,
                    description=entity.description,
                    aliases=entity.aliases,
                )

    def add_relationship(self, rel: Relationship):
        """Add a relationship edge to the graph."""
        source = self._resolve_name(rel.source) or rel.source
        target = self._resolve_name(rel.target) or rel.target

        # Ensure both nodes exist
        if source not in self.graph.nodes:
            self.graph.add_node(source, entity_type="unknown", description="", aliases=[])
        if target not in self.graph.nodes:
            self.graph.add_node(target, entity_type="unknown", description="", aliases=[])

        self.graph.add_edge(
            source, target,
            relation_type=rel.relation_type,
            evidence=rel.evidence,
            confidence=rel.confidence,
        )

    def _resolve_name(self, name: str) -> str | None:
        """Resolve a name to its canonical node name (handling aliases)."""
        # Exact match
        if name in self.graph.nodes:
            return name

        # Check aliases
        name_lower = name.lower()
        for node, data in self.graph.nodes(data=True):
            if node.lower() == name_lower:
                return node
            aliases = data.get("aliases", [])
            if any(a.lower() == name_lower for a in aliases):
                return node

        return None

    def get_neighbours(self, node_name: str, depth: int = 2) -> dict[str, Any]:
        """Get all neighbours within N hops of a node."""
        resolved = self._resolve_name(node_name)
        if not resolved:
            return {"error": f"Node '{node_name}' not found in graph"}

        result = {
            "center": resolved,
            "center_data": dict(self.graph.nodes[resolved]),
            "outgoing": [],
            "incoming": [],
            "paths": [],
        }

        # Outgoing edges
        for _, target, data in self.graph.out_edges(resolved, data=True):
            result["outgoing"].append({
                "target": target,
                "relation": data.get("relation_type", "related_to"),
                "confidence": data.get("confidence", 0.0),
            })

        # Incoming edges
        for source, _, data in self.graph.in_edges(resolved, data=True):
            result["incoming"].append({
                "source": source,
                "relation": data.get("relation_type", "related_to"),
                "confidence": data.get("confidence", 0.0),
            })

        # Multi-hop paths (depth-limited BFS)
        if depth >= 2:
            visited = {resolved}
            frontier = [(resolved, [])]
            for _ in range(depth):
                next_frontier = []
                for current, path in frontier:
                    for _, nbr, data in self.graph.out_edges(current, data=True):
                        if nbr not in visited:
                            visited.add(nbr)
                            new_path = path + [{"from": current, "to": nbr, "rel": data.get("relation_type", "")}]
                            result["paths"].append(new_path)
                            next_frontier.append((nbr, new_path))
                    for nbr, _, data in self.graph.in_edges(current, data=True):
                        if nbr not in visited:
                            visited.add(nbr)
                            new_path = path + [{"from": nbr, "to": current, "rel": data.get("relation_type", "")}]
                            result["paths"].append(new_path)
                            next_frontier.append((nbr, new_path))
                frontier = next_frontier

        return result

    def find_path(self, source_name: str, target_name: str) -> list[dict]:
        """Find shortest path between two entities."""
        src = self._resolve_name(source_name)
        tgt = self._resolve_name(target_name)

        if not src or not tgt:
            return []

        try:
            # Try undirected shortest path
            undirected = self.graph.to_undirected()
            path_nodes = nx.shortest_path(undirected, src, tgt)

            path_steps = []
            for i in range(len(path_nodes) - 1):
                a, b = path_nodes[i], path_nodes[i + 1]
                # Check directed edge
                if self.graph.has_edge(a, b):
                    edge_data = self.graph.edges[a, b]
                    rel = edge_data.get("relation_type", "related_to")
                elif self.graph.has_edge(b, a):
                    edge_data = self.graph.edges[b, a]
                    rel = f"(reverse) {edge_data.get('relation_type', 'related_to')}"
                else:
                    rel = "connected_to"
                path_steps.append({"from": a, "to": b, "relation": rel})

            return path_steps
        except nx.NetworkXNoPath:
            return []
        except nx.NodeNotFound:
            return []

    def stats(self) -> dict:
        """Get graph statistics."""
        node_types: dict[str, int] = {}
        for _, data in self.graph.nodes(data=True):
            etype = data.get("entity_type", "unknown")
            node_types[etype] = node_types.get(etype, 0) + 1

        rel_types: dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            rtype = data.get("relation_type", "related_to")
            rel_types[rtype] = rel_types.get(rtype, 0) + 1

        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "node_types": node_types,
            "relationship_types": rel_types,
        }


# ═══════════════════════════════════════════════════════════════════════════
#  PART 5 — QUERY AGENT (Natural Language → Graph Traversal)
# ═══════════════════════════════════════════════════════════════════════════

class QueryAgent:
    """Translates natural language questions into graph traversals."""

    def __init__(self, kg: KnowledgeGraph):
        self.kg = kg

    def answer(self, question: str) -> QueryResult:
        """Answer a question by traversing the knowledge graph."""
        # Step 1: Identify entities mentioned in the question
        entities_in_question = self._extract_question_entities(question)

        # Step 2: Gather graph context for all relevant entities
        graph_context = self._gather_context(entities_in_question)

        # Step 3: Use LLM to answer based on graph context
        answer, traversal_path, nodes_visited = self._generate_answer(question, graph_context)

        return QueryResult(
            question=question,
            answer=answer,
            traversal_path=traversal_path,
            confidence=0.8 if traversal_path else 0.4,
            hops=len(traversal_path),
            nodes_visited=nodes_visited,
        )

    def _extract_question_entities(self, question: str) -> list[str]:
        """Extract entity names from a question using graph node matching + LLM."""
        # First: direct substring matching against known nodes
        found = []
        q_lower = question.lower()
        for node in self.kg.graph.nodes:
            if node.lower() in q_lower:
                found.append(node)
            else:
                # Check aliases
                aliases = self.kg.graph.nodes[node].get("aliases", [])
                for alias in aliases:
                    if alias.lower() in q_lower:
                        found.append(node)
                        break

        # If no direct matches, use LLM
        if not found:
            all_nodes = list(self.kg.graph.nodes)
            result = llm_call(
                "You are an entity matcher. Given a question and a list of known entities, "
                "identify which entities are relevant. Return ONLY a JSON list of entity names, e.g. [\"Entity A\", \"Entity B\"].",
                f"Question: {question}\n\nKnown entities: {json.dumps(all_nodes[:100])}",
                temperature=0.1,
            )
            try:
                cleaned = result.strip()
                if cleaned.startswith("```"):
                    cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
                    cleaned = cleaned.rstrip("`").strip()
                matched = json.loads(cleaned)
                found = [e for e in matched if e in self.kg.graph.nodes]
            except (json.JSONDecodeError, TypeError):
                pass

        return found

    def _gather_context(self, entities: list[str]) -> str:
        """Gather graph context for a list of entities."""
        context_parts = []

        for entity in entities:
            neighbours = self.kg.get_neighbours(entity, depth=2)
            if "error" in neighbours:
                continue

            entity_data = neighbours.get("center_data", {})
            context_parts.append(f"\n=== Entity: {entity} ===")
            context_parts.append(f"Type: {entity_data.get('entity_type', 'unknown')}")
            if entity_data.get("description"):
                context_parts.append(f"Description: {entity_data['description']}")

            if neighbours.get("outgoing"):
                context_parts.append("Outgoing relationships:")
                for rel in neighbours["outgoing"]:
                    context_parts.append(f"  {entity} --[{rel['relation']}]--> {rel['target']}")

            if neighbours.get("incoming"):
                context_parts.append("Incoming relationships:")
                for rel in neighbours["incoming"]:
                    context_parts.append(f"  {rel['source']} --[{rel['relation']}]--> {entity}")

            if neighbours.get("paths"):
                context_parts.append("Multi-hop paths:")
                for path in neighbours["paths"]:
                    path_str = " → ".join([f"{step['from']} --[{step['rel']}]--> {step['to']}" for step in path])
                    context_parts.append(f"  {path_str}")

        # Also find paths between entities if multiple
        if len(entities) >= 2:
            for i in range(len(entities)):
                for j in range(i + 1, len(entities)):
                    path = self.kg.find_path(entities[i], entities[j])
                    if path:
                        context_parts.append(f"\nPath from {entities[i]} to {entities[j]}:")
                        for step in path:
                            context_parts.append(f"  {step['from']} --[{step['relation']}]--> {step['to']}")

        return "\n".join(context_parts) if context_parts else "No relevant graph context found."

    def _generate_answer(self, question: str, graph_context: str) -> tuple[str, list[GraphTraversalStep], list[str]]:
        """Generate an answer from the graph context."""
        system_prompt = """You are a knowledge graph query agent. Answer the question using ONLY the graph context provided.
Your answer must:
1. Be based entirely on the relationships and entities in the graph
2. Show the reasoning path (which edges you traversed)
3. Be specific and factual

Format your response as:
ANSWER: [your answer]
PATH: [entity1] --[relation]--> [entity2] --[relation]--> [entity3]
NODES: [comma-separated list of nodes visited]

If the graph context doesn't contain enough information, say so explicitly."""

        response = llm_call(
            system_prompt,
            f"Question: {question}\n\nGraph Context:\n{graph_context}",
            temperature=0.0,
        )

        # Parse the structured response
        answer = response
        traversal_path: list[GraphTraversalStep] = []
        nodes_visited: list[str] = []

        if "ANSWER:" in response:
            parts = response.split("ANSWER:", 1)[1]
            if "PATH:" in parts:
                answer = parts.split("PATH:", 1)[0].strip()
                path_part = parts.split("PATH:", 1)[1]
                if "NODES:" in path_part:
                    path_str = path_part.split("NODES:", 1)[0].strip()
                    nodes_str = path_part.split("NODES:", 1)[1].strip()
                    nodes_visited = [n.strip() for n in nodes_str.split(",") if n.strip()]
                else:
                    path_str = path_part.strip()

                # Parse path string into GraphTraversalStep objects
                edge_pattern = re.findall(r"(.+?)\s*--\[(.+?)\]-->\s*", path_str)
                for i, (node, edge_type) in enumerate(edge_pattern):
                    next_node = ""
                    if i + 1 < len(edge_pattern):
                        next_node = edge_pattern[i + 1][0].strip()
                    elif edge_pattern:
                        # Last edge — get the final node
                        remaining = path_str.split(f"--[{edge_type}]-->")[-1].strip()
                        # Clean remaining of any further arrows
                        next_node = remaining.split("--[")[0].strip() if "--[" in remaining else remaining.strip()
                    traversal_path.append(GraphTraversalStep(
                        node=node.strip(),
                        edge_type=edge_type.strip(),
                        next_node=next_node,
                    ))
            else:
                answer = parts.strip()

        return answer, traversal_path, nodes_visited


# ═══════════════════════════════════════════════════════════════════════════
#  PART 6 — PYVIS VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════

def generate_pyvis_graph(kg: KnowledgeGraph, output_path: str = str(PYVIS_PATH)):
    """Generate an interactive pyvis HTML visualisation of the knowledge graph."""
    try:
        from pyvis.network import Network
    except ImportError:
        console.print("[yellow]pyvis not installed. Install with: pip install pyvis[/]")
        # Fallback: generate a simple HTML visualisation
        _generate_simple_html(kg, output_path)
        return

    net = Network(
        height="800px",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True,
        notebook=False,
    )

    # Color map for entity types
    color_map = {
        "person": "#e74c3c",
        "company": "#3498db",
        "technology": "#2ecc71",
        "concept": "#f39c12",
        "place": "#9b59b6",
        "product": "#1abc9c",
        "unknown": "#95a5a6",
    }

    # Add nodes
    for node, data in kg.graph.nodes(data=True):
        entity_type = data.get("entity_type", "unknown")
        color = color_map.get(entity_type, "#95a5a6")
        title = f"Type: {entity_type}\nDescription: {data.get('description', 'N/A')}"
        if data.get("aliases"):
            title += f"\nAliases: {', '.join(data['aliases'])}"

        # Size based on degree centrality
        degree = kg.graph.degree(node)
        size = max(15, min(50, 10 + degree * 5))

        net.add_node(
            node,
            label=node,
            color=color,
            size=size,
            title=title,
            font={"size": 12},
        )

    # Add edges
    for source, target, data in kg.graph.edges(data=True):
        rel_type = data.get("relation_type", "related_to")
        confidence = data.get("confidence", 0.8)
        net.add_edge(
            source, target,
            label=rel_type,
            title=f"{rel_type} (confidence: {confidence:.2f})",
            color="#555555",
            width=max(1, confidence * 3),
        )

    # Physics settings for better layout
    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 150,
          "springConstant": 0.08
        },
        "solver": "forceAtlas2Based",
        "stabilization": {"iterations": 150}
      },
      "interaction": {
        "hover": true,
        "navigationButtons": true
      }
    }
    """)

    net.save_graph(output_path)
    console.print(f"[green]Graph visualisation saved to {output_path}[/]")


def _generate_simple_html(kg: KnowledgeGraph, output_path: str):
    """Fallback: simple HTML table visualisation when pyvis is unavailable."""
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Knowledge Graph - Lab 6.2</title>
<style>
body { font-family: 'Segoe UI', sans-serif; background: #1a1a2e; color: #e0e0e0; padding: 20px; }
h1 { color: #3498db; }
table { border-collapse: collapse; width: 100%; margin-bottom: 30px; }
th { background: #16213e; color: #3498db; padding: 10px; text-align: left; }
td { padding: 8px; border-bottom: 1px solid #333; }
.entity-person { color: #e74c3c; } .entity-company { color: #3498db; }
.entity-technology { color: #2ecc71; } .entity-concept { color: #f39c12; }
.entity-place { color: #9b59b6; } .entity-product { color: #1abc9c; }
</style></head><body>
<h1>Knowledge Graph — Lab 6.2</h1>
"""
    html += f"<p>Nodes: {kg.graph.number_of_nodes()} | Edges: {kg.graph.number_of_edges()}</p>"
    html += "<h2>Entities</h2><table><tr><th>Name</th><th>Type</th><th>Description</th></tr>"
    for node, data in kg.graph.nodes(data=True):
        etype = data.get("entity_type", "unknown")
        html += f"<tr><td class='entity-{etype}'>{node}</td><td>{etype}</td><td>{data.get('description', '')}</td></tr>"
    html += "</table>"
    html += "<h2>Relationships</h2><table><tr><th>Source</th><th>Relation</th><th>Target</th><th>Confidence</th></tr>"
    for src, tgt, data in kg.graph.edges(data=True):
        html += f"<tr><td>{src}</td><td>{data.get('relation_type', '')}</td><td>{tgt}</td><td>{data.get('confidence', 0):.2f}</td></tr>"
    html += "</table></body></html>"

    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    console.print(f"[green]HTML visualisation saved to {output_path}[/]")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 7 — CLI COMMANDS
# ═══════════════════════════════════════════════════════════════════════════

@app.command()
def ingest():
    """Ingest the 20-paragraph corpus and build the knowledge graph."""
    console.print(Rule("[bold cyan]Lab 6.2 — Knowledge Graph Ingestion[/]"))
    console.print(f"\n[bold]Corpus:[/] {len(CORPUS)} paragraphs (AI/Tech domain)")
    console.print()

    kg = KnowledgeGraph()
    extractor = EntityRelationExtractor()

    total_entities = 0
    total_relationships = 0

    for i, paragraph in enumerate(CORPUS):
        console.print(f"[cyan]Processing paragraph {i + 1}/{len(CORPUS)}...[/]")
        result = extractor.extract(paragraph, paragraph_index=i)

        for entity in result.entities:
            kg.add_entity(entity)
            total_entities += 1

        for rel in result.relationships:
            kg.add_relationship(rel)
            total_relationships += 1

        console.print(f"  [dim]Extracted {len(result.entities)} entities, {len(result.relationships)} relationships[/]")
        time.sleep(2)  # Rate limit spacing

    kg.save()

    # Stats
    console.print()
    console.print(Rule("[bold green]Ingestion Complete[/]"))
    stats_table = Table(title="Knowledge Graph Statistics", box=box.ROUNDED)
    stats_table.add_column("Metric", style="bold")
    stats_table.add_column("Value", style="cyan")
    stats_table.add_row("Paragraphs processed", str(len(CORPUS)))
    stats_table.add_row("Entities extracted", str(total_entities))
    stats_table.add_row("Relationships extracted", str(total_relationships))
    stats_table.add_row("Unique graph nodes", str(kg.graph.number_of_nodes()))
    stats_table.add_row("Unique graph edges", str(kg.graph.number_of_edges()))
    console.print(stats_table)


@app.command()
def query(question: str = typer.Argument(..., help="Natural language question to answer")):
    """Query the knowledge graph with a natural language question."""
    console.print(Rule("[bold cyan]Lab 6.2 — Knowledge Graph Query[/]"))
    console.print()

    kg = KnowledgeGraph()
    if kg.graph.number_of_nodes() == 0:
        console.print("[red]Graph is empty. Run 'ingest' first.[/]")
        raise typer.Exit(1)

    agent = QueryAgent(kg)
    result = agent.answer(question)

    console.print(Panel(
        f"[bold]Question:[/] {result.question}\n\n"
        f"[bold green]Answer:[/] {result.answer}\n\n"
        f"[bold]Hops:[/] {result.hops}\n"
        f"[bold]Confidence:[/] {result.confidence:.2f}\n"
        f"[bold]Nodes visited:[/] {', '.join(result.nodes_visited) if result.nodes_visited else 'N/A'}",
        title="[bold cyan]Query Result[/]",
        border_style="cyan",
    ))

    if result.traversal_path:
        path_tree = Tree("[bold]Traversal Path[/]")
        for step in result.traversal_path:
            path_tree.add(f"[cyan]{step.node}[/] --[[yellow]{step.edge_type}[/]]--> [green]{step.next_node}[/]")
        console.print(path_tree)


@app.command()
def visualise():
    """Generate an interactive pyvis HTML graph visualisation."""
    console.print(Rule("[bold cyan]Lab 6.2 — Graph Visualisation[/]"))

    kg = KnowledgeGraph()
    if kg.graph.number_of_nodes() == 0:
        console.print("[red]Graph is empty. Run 'ingest' first.[/]")
        raise typer.Exit(1)

    generate_pyvis_graph(kg)


@app.command()
def validate():
    """Run the 5 multi-hop validation questions."""
    console.print(Rule("[bold cyan]Lab 6.2 — Multi-Hop Query Validation[/]"))
    console.print()

    kg = KnowledgeGraph()
    if kg.graph.number_of_nodes() == 0:
        console.print("[yellow]Graph is empty. Running ingestion first...[/]\n")
        ingest()
        kg = KnowledgeGraph()

    agent = QueryAgent(kg)

    # 5 multi-hop questions (each requires crossing 2+ edges)
    multi_hop_questions = [
        {
            "question": "Who founded the company that developed GPT-4?",
            "expected_keywords": ["sam altman", "elon musk", "greg brockman", "ilya sutskever"],
            "explanation": "Requires: GPT-4 → developed_by → OpenAI → founded_by → [founders]",
            "hops_required": 2,
        },
        {
            "question": "Who is the CEO of the company that developed AlphaFold?",
            "expected_keywords": ["demis hassabis"],
            "explanation": "Requires: AlphaFold → developed_by → Google DeepMind → ceo_of → Demis Hassabis",
            "hops_required": 2,
        },
        {
            "question": "Where is the headquarters of the company that originally developed PyTorch?",
            "expected_keywords": ["menlo park"],
            "explanation": "Requires: PyTorch → developed_by → Meta AI → headquartered_in → Menlo Park",
            "hops_required": 2,
        },
        {
            "question": "Which cloud provider invested heavily in the startup that developed the Claude family of models?",
            "expected_keywords": ["aws", "amazon web services", "amazon"],
            "explanation": "Requires: Claude → developed_by → Anthropic → invested_in_by → AWS",
            "hops_required": 2,
        },
        {
            "question": "What startup accelerator was previously led by the CEO of the organisation that released and developed GPT-4?",
            "expected_keywords": ["y combinator", "president"],
            "explanation": "Requires: GPT-4 → developed_by → OpenAI → ceo → Sam Altman → previous role → Y Combinator",
            "hops_required": 2,
        },
    ]

    # Also test keyword search failure
    keyword_failures = []

    results = {"passed": 0, "failed": 0, "details": []}
    results_table = Table(title="Multi-Hop Query Validation", box=box.ROUNDED)
    results_table.add_column("#", style="dim", width=3)
    results_table.add_column("Question", width=45)
    results_table.add_column("Graph QA", justify="center", width=10)
    results_table.add_column("Keyword Fail?", justify="center", width=12)

    for i, test in enumerate(multi_hop_questions, 1):
        console.print(f"\n[bold]Question {i}:[/] {test['question']}")
        console.print(f"  [dim]{test['explanation']}[/]")

        # Graph-based answer
        result = agent.answer(test["question"])
        console.print(f"  [green]Graph answer:[/] {result.answer[:150]}")

        # Check answer
        answer_lower = result.answer.lower()
        found = [kw for kw in test["expected_keywords"] if kw in answer_lower]
        passed = len(found) >= 1

        # Keyword search test (should fail for multi-hop)
        keyword_result = _keyword_search(test["question"])
        keyword_found = any(kw in keyword_result.lower() for kw in test["expected_keywords"])

        if not keyword_found:
            keyword_failures.append(i)

        if passed:
            results["passed"] += 1
        else:
            results["failed"] += 1

        results["details"].append({
            "question": test["question"],
            "answer": result.answer,
            "passed": passed,
            "keyword_search_failed": not keyword_found,
        })

        graph_status = "[green]PASS[/]" if passed else "[red]FAIL[/]"
        kw_status = "[green]Yes (fail)[/]" if not keyword_found else "[yellow]No[/]"
        results_table.add_row(str(i), test["question"][:45], graph_status, kw_status)
        time.sleep(2)

    console.print()
    console.print(results_table)

    total = results["passed"] + results["failed"]
    console.print(f"\n[bold]Graph QA Score:[/] {results['passed']}/{total} (need ≥4)")
    console.print(f"[bold]Keyword Search Failures:[/] {len(keyword_failures)}/{total} (need ≥2)")

    if results["passed"] >= 4 and len(keyword_failures) >= 2:
        console.print("[bold green]✓ Validation PASSED[/]")
    else:
        console.print("[bold red]✗ Validation FAILED[/]")

    # Save report
    report_path = ROOT_DIR / "labs" / "lab_6_2_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"\n[dim]Report saved to {report_path}[/]")


def _keyword_search(question: str) -> str:
    """Simple keyword search over the corpus (baseline comparison)."""
    import string
    question_clean = question.translate(str.maketrans('', '', string.punctuation))
    question_words = set(question_clean.lower().split())
    # Remove stop words
    stop_words = {"who", "what", "where", "when", "how", "is", "the", "a", "an", "of", "in", "at", "by", "that", "did", "does", "which"}
    keywords = question_words - stop_words

    best_score = 0
    best_paragraph = ""
    for paragraph in CORPUS:
        para_lower = paragraph.lower()
        score = sum(1 for kw in keywords if kw in para_lower)
        if score > best_score:
            best_score = score
            best_paragraph = paragraph

    return best_paragraph


@app.command()
def demo():
    """Run the full demo: ingest, query, and visualise."""
    console.print(Rule("[bold cyan]Lab 6.2 — Knowledge Graph Query Agent — Full Demo[/]"))
    console.print()
    console.print(Panel(
        "[bold]This demo will:[/]\n"
        "1. Ingest 20 paragraphs and build a knowledge graph\n"
        "2. Answer 3 multi-hop questions with graph traversal\n"
        "3. Generate an interactive HTML visualisation",
        title="[bold cyan]Demo Overview[/]",
        border_style="cyan",
    ))
    console.print()

    # Step 1: Ingest
    ingest()
    time.sleep(2)

    # Step 2: Query examples
    console.print()
    console.print(Rule("[bold green]Multi-Hop Query Demonstrations[/]"))

    kg = KnowledgeGraph()
    agent = QueryAgent(kg)

    demo_questions = [
        "Who founded the company that developed GPT-4?",
        "Who is the CEO of the organisation that created AlphaFold?",
        "What framework is used to build the models developed by Meta AI?",
    ]

    for q in demo_questions:
        console.print(f"\n[bold blue]Question:[/] {q}")
        result = agent.answer(q)
        console.print(f"[bold green]Answer:[/] {result.answer}")
        if result.nodes_visited:
            console.print(f"[dim]Nodes visited: {', '.join(result.nodes_visited)}[/]")
        console.print(f"[dim]Hops: {result.hops} | Confidence: {result.confidence:.2f}[/]")
        time.sleep(2)

    # Step 3: Visualise
    console.print()
    generate_pyvis_graph(kg)

    console.print()
    console.print(Rule("[bold cyan]Demo Complete[/]"))
    console.print(f"\n[bold]Graph:[/] {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges")
    console.print(f"[bold]Visualisation:[/] {PYVIS_PATH}")


if __name__ == "__main__":
    app()
