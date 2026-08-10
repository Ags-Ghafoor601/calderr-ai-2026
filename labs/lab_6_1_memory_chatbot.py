#!/usr/bin/env python3
"""
CalderR Internship – Week 6, Lab 6.1
======================================
Memory-Augmented Chatbot — Cross-Session Recall with Dual Memory

WHAT THIS LAB BUILDS:
---------------------
A CLI chatbot backed by two persistent memory stores:
  • SQLite episodic store: raw interaction history with timestamps,
    session IDs, importance scores, and user IDs
  • ChromaDB semantic index: embedded summaries of past sessions
    for relevance-based retrieval
  • On every new session the agent queries BOTH stores and injects
    the 5 most relevant past memories into context before responding
  • Recency + relevance blending ensures recent important memories
    surface alongside semantically similar older ones

WHAT THIS TEACHES YOU:
----------------------
  • Episodic vs semantic memory — when to use each
  • SQLite for structured event logs with timestamps
  • ChromaDB for embedding-based similarity retrieval
  • Recency-relevance blending: scoring memories by both freshness
    and semantic similarity to the current query
  • Cross-session persistence — memory survives app restarts
  • Memory injection into LLM context windows

ARCHITECTURE:
    ┌─────────────────────────────────────────────────────────┐
    │                  MEMORY-AUGMENTED CHATBOT                │
    │                                                         │
    │  ┌──────────────┐        ┌───────────────┐              │
    │  │ SQLite       │        │ ChromaDB      │              │
    │  │ Episodic     │        │ Semantic      │              │
    │  │ Store        │        │ Index         │              │
    │  │ ────────     │        │ ────────      │              │
    │  │ timestamp    │        │ embedded      │              │
    │  │ session_id   │        │ summaries     │              │
    │  │ user_msg     │        │ of past       │              │
    │  │ assistant_msg│        │ sessions      │              │
    │  │ importance   │        │               │              │
    │  └──────┬───────┘        └───────┬───────┘              │
    │         │    ┌───────────────┐   │                      │
    │         └───►│  MEMORY       │◄──┘                      │
    │              │  RETRIEVER    │                           │
    │              │  (recency +   │                           │
    │              │   relevance   │                           │
    │              │   blending)   │                           │
    │              └───────┬──────┘                            │
    │                      │  Top-5 memories                  │
    │              ┌───────▼──────┐                            │
    │              │  GROQ LLM   │                            │
    │              │  (context-   │                            │
    │              │   augmented) │                            │
    │              └──────────────┘                            │
    └─────────────────────────────────────────────────────────┘

    Flow:
    1. User starts a new session
    2. Memory Retriever queries SQLite (recency) + ChromaDB (relevance)
    3. Top-5 blended memories injected into system prompt
    4. User chats with the agent
    5. Every exchange logged to SQLite + ChromaDB updated on session end
    6. Session 3 can recall facts from session 1

Run:
    python labs/lab_6_1_memory_chatbot.py demo
    python labs/lab_6_1_memory_chatbot.py session --user alice
    python labs/lab_6_1_memory_chatbot.py validate
    python labs/lab_6_1_memory_chatbot.py inspect --user alice
"""

# pylint: disable=line-too-long, too-many-locals, wrong-import-position, broad-exception-caught, missing-class-docstring, missing-function-docstring, too-few-public-methods, duplicate-code
import io
import os
import sys
import json
import time
import uuid
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional, Any

# Fix Windows console encoding
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import typer
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
app = typer.Typer(help="Lab 6.1 — Memory-Augmented Chatbot with Cross-Session Recall")

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL = "llama-3.1-8b-instant"
DB_PATH = ROOT_DIR / "labs" / ".memory_lab61.db"
CHROMA_PATH = str(ROOT_DIR / "labs" / ".chromadb_lab61")


# ─── LLM Helper ────────────────────────────────────────────────────────────
def llm_call(system_prompt: str, user_prompt: str, temperature: float = 0.7) -> str:
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
                max_tokens=1024,
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
#  PART 1 — PYDANTIC MODELS
# ═══════════════════════════════════════════════════════════════════════════

class EpisodicMemory(BaseModel):
    """A single episodic memory entry — one turn of conversation."""
    memory_id: str = Field(default_factory=lambda: str(uuid.uuid4())[:8])
    session_id: str = Field(..., description="Session this memory belongs to")
    user_id: str = Field(default="default", description="User who created this memory")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    user_message: str = Field(..., description="What the user said")
    assistant_response: str = Field(..., description="What the assistant replied")
    importance_score: float = Field(default=0.5, ge=0.0, le=1.0, description="0=trivial, 1=critical")
    topic: str = Field(default="general", description="Extracted topic of this exchange")


class SessionSummary(BaseModel):
    """Summary of an entire session — stored in ChromaDB for semantic search."""
    session_id: str = Field(...)
    user_id: str = Field(default="default")
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    summary: str = Field(..., description="LLM-generated summary of the session")
    key_topics: list[str] = Field(default_factory=list, description="Main topics discussed")
    num_exchanges: int = Field(default=0, description="Number of user-assistant exchanges")


class RetrievedMemory(BaseModel):
    """A memory retrieved for context injection — with relevance scores."""
    source: str = Field(..., description="'episodic' or 'semantic'")
    content: str = Field(..., description="The memory content")
    relevance_score: float = Field(default=0.0, description="Semantic similarity score")
    recency_score: float = Field(default=0.0, description="Time-based recency score")
    blended_score: float = Field(default=0.0, description="Combined relevance + recency")
    session_id: str = Field(default="")
    timestamp: str = Field(default="")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 2 — SQLITE EPISODIC STORE
# ═══════════════════════════════════════════════════════════════════════════

class EpisodicStore:
    """SQLite-backed episodic memory store with timestamps and importance scoring."""

    def __init__(self, db_path: str = str(DB_PATH)):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        """Create the episodic_memories table if it doesn't exist."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            CREATE TABLE IF NOT EXISTS episodic_memories (
                memory_id TEXT PRIMARY KEY,
                session_id TEXT NOT NULL,
                user_id TEXT NOT NULL DEFAULT 'default',
                timestamp TEXT NOT NULL,
                user_message TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                importance_score REAL NOT NULL DEFAULT 0.5,
                topic TEXT NOT NULL DEFAULT 'general'
            )
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodic_user
            ON episodic_memories(user_id, timestamp DESC)
        """)
        conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_episodic_session
            ON episodic_memories(session_id)
        """)
        conn.commit()
        conn.close()

    def store(self, memory: EpisodicMemory):
        """Store a single episodic memory."""
        conn = sqlite3.connect(self.db_path)
        conn.execute("""
            INSERT OR REPLACE INTO episodic_memories
            (memory_id, session_id, user_id, timestamp, user_message,
             assistant_response, importance_score, topic)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            memory.memory_id, memory.session_id, memory.user_id,
            memory.timestamp, memory.user_message, memory.assistant_response,
            memory.importance_score, memory.topic,
        ))
        conn.commit()
        conn.close()

    def get_recent(self, user_id: str, limit: int = 10) -> list[EpisodicMemory]:
        """Retrieve most recent memories for a user, ordered by timestamp descending."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT memory_id, session_id, user_id, timestamp,
                   user_message, assistant_response, importance_score, topic
            FROM episodic_memories
            WHERE user_id = ?
            ORDER BY timestamp DESC
            LIMIT ?
        """, (user_id, limit)).fetchall()
        conn.close()
        return [
            EpisodicMemory(
                memory_id=r[0], session_id=r[1], user_id=r[2], timestamp=r[3],
                user_message=r[4], assistant_response=r[5],
                importance_score=r[6], topic=r[7],
            )
            for r in rows
        ]

    def get_by_session(self, session_id: str) -> list[EpisodicMemory]:
        """Retrieve all memories for a given session."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT memory_id, session_id, user_id, timestamp,
                   user_message, assistant_response, importance_score, topic
            FROM episodic_memories
            WHERE session_id = ?
            ORDER BY timestamp ASC
        """, (session_id,)).fetchall()
        conn.close()
        return [
            EpisodicMemory(
                memory_id=r[0], session_id=r[1], user_id=r[2], timestamp=r[3],
                user_message=r[4], assistant_response=r[5],
                importance_score=r[6], topic=r[7],
            )
            for r in rows
        ]

    def get_all_sessions(self, user_id: str) -> list[str]:
        """Get all unique session IDs for a user, ordered by most recent."""
        conn = sqlite3.connect(self.db_path)
        rows = conn.execute("""
            SELECT DISTINCT session_id
            FROM episodic_memories
            WHERE user_id = ?
            ORDER BY timestamp DESC
        """, (user_id,)).fetchall()
        conn.close()
        return [r[0] for r in rows]

    def count_memories(self, user_id: str) -> int:
        """Count total memories for a user."""
        conn = sqlite3.connect(self.db_path)
        count = conn.execute(
            "SELECT COUNT(*) FROM episodic_memories WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0]
        conn.close()
        return count


# ═══════════════════════════════════════════════════════════════════════════
#  PART 3 — CHROMADB SEMANTIC INDEX
# ═══════════════════════════════════════════════════════════════════════════

class SemanticIndex:
    """ChromaDB-backed semantic memory for session summaries and key facts."""

    def __init__(self, persist_dir: str = CHROMA_PATH):
        import chromadb
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(
            name="session_summaries",
            metadata={"hnsw:space": "cosine"},
        )

    def store_summary(self, summary: SessionSummary):
        """Embed and store a session summary for semantic retrieval."""
        doc_text = f"Session summary for {summary.user_id}: {summary.summary}"
        if summary.key_topics:
            doc_text += f" Key topics: {', '.join(summary.key_topics)}"
        self.collection.upsert(
            ids=[summary.session_id],
            documents=[doc_text],
            metadatas=[{
                "user_id": summary.user_id,
                "session_id": summary.session_id,
                "timestamp": summary.timestamp,
                "key_topics": json.dumps(summary.key_topics),
                "num_exchanges": summary.num_exchanges,
            }],
        )

    def query(self, query_text: str, user_id: str, n_results: int = 5) -> list[dict]:
        """Query the semantic index for relevant session summaries."""
        try:
            results = self.collection.query(
                query_texts=[query_text],
                n_results=n_results,
                where={"user_id": user_id},
            )
        except Exception:
            # If filtering fails (empty collection), try without filter
            try:
                results = self.collection.query(
                    query_texts=[query_text],
                    n_results=n_results,
                )
            except Exception:
                return []

        memories = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                distance = results["distances"][0][i] if results["distances"] else 1.0
                # ChromaDB returns distance; convert to similarity
                similarity = max(0.0, 1.0 - distance)
                memories.append({
                    "content": doc,
                    "session_id": meta.get("session_id", ""),
                    "timestamp": meta.get("timestamp", ""),
                    "similarity": similarity,
                })
        return memories

    def count(self) -> int:
        """Count total stored summaries."""
        return self.collection.count()


# ═══════════════════════════════════════════════════════════════════════════
#  PART 4 — MEMORY RETRIEVER (RECENCY + RELEVANCE BLENDING)
# ═══════════════════════════════════════════════════════════════════════════

class MemoryRetriever:
    """Blends episodic (recency-weighted) and semantic (relevance-weighted) memories."""

    def __init__(self, episodic_store: EpisodicStore, semantic_index: SemanticIndex):
        self.episodic = episodic_store
        self.semantic = semantic_index
        self.recency_weight = 0.4
        self.relevance_weight = 0.6

    def _compute_recency_score(self, timestamp_str: str) -> float:
        """Compute recency score: 1.0 for now, decaying over time."""
        try:
            ts = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            age_hours = max(0, (now - ts).total_seconds() / 3600)
            # Exponential decay: half-life of 168 hours (1 week)
            return 2.0 ** (-age_hours / 168.0)
        except Exception:
            return 0.5

    def retrieve(self, query: str, user_id: str, top_k: int = 5) -> list[RetrievedMemory]:
        """Retrieve top-k memories by blending recency and relevance."""
        candidates: list[RetrievedMemory] = []

        # 1. Episodic memories (recency-weighted)
        recent_memories = self.episodic.get_recent(user_id, limit=20)
        for mem in recent_memories:
            recency = self._compute_recency_score(mem.timestamp)
            # Importance boosts recency
            adjusted_recency = recency * (0.5 + 0.5 * mem.importance_score)
            content = f"[Session {mem.session_id[:6]}] User asked: {mem.user_message} | Agent replied: {mem.assistant_response}"
            candidates.append(RetrievedMemory(
                source="episodic",
                content=content,
                relevance_score=0.0,  # No semantic similarity for episodic
                recency_score=adjusted_recency,
                blended_score=self.recency_weight * adjusted_recency,
                session_id=mem.session_id,
                timestamp=mem.timestamp,
            ))

        # 2. Semantic memories (relevance-weighted)
        semantic_results = self.semantic.query(query, user_id, n_results=10)
        for result in semantic_results:
            recency = self._compute_recency_score(result.get("timestamp", ""))
            relevance = result.get("similarity", 0.0)
            blended = (self.relevance_weight * relevance) + (self.recency_weight * recency)
            candidates.append(RetrievedMemory(
                source="semantic",
                content=result["content"],
                relevance_score=relevance,
                recency_score=recency,
                blended_score=blended,
                session_id=result.get("session_id", ""),
                timestamp=result.get("timestamp", ""),
            ))

        # 3. Sort by blended score and deduplicate by session
        candidates.sort(key=lambda m: m.blended_score, reverse=True)
        seen_sessions: set[str] = set()
        top_memories: list[RetrievedMemory] = []
        for mem in candidates:
            key = mem.session_id or mem.content[:50]
            if key not in seen_sessions:
                seen_sessions.add(key)
                top_memories.append(mem)
            if len(top_memories) >= top_k:
                break

        return top_memories


# ═══════════════════════════════════════════════════════════════════════════
#  PART 5 — IMPORTANCE SCORER
# ═══════════════════════════════════════════════════════════════════════════

def score_importance(user_message: str, assistant_response: str) -> float:
    """Use LLM to score the importance of an interaction (0.0 to 1.0)."""
    prompt = f"""Rate the importance of this conversation exchange on a scale of 0.0 to 1.0.
Consider: Does it contain personal information, preferences, key facts, decisions,
or commitments? Generic chit-chat = 0.1-0.3. Personal facts = 0.5-0.7.
Important decisions/preferences = 0.8-1.0.

User: {user_message}
Assistant: {assistant_response}

Respond with ONLY a decimal number between 0.0 and 1.0, nothing else."""

    result = llm_call(
        "You are an importance scorer. Output ONLY a decimal number between 0.0 and 1.0.",
        prompt,
        temperature=0.1,
    )
    try:
        score = float(result.strip())
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError):
        return 0.5


def extract_topic(user_message: str) -> str:
    """Extract the main topic from a user message."""
    result = llm_call(
        "You are a topic extractor. Output ONLY a 1-3 word topic label, nothing else.",
        f"Extract the main topic from this message: {user_message}",
        temperature=0.1,
    )
    return result.strip()[:50] if result else "general"


def summarise_session(exchanges: list[EpisodicMemory]) -> str:
    """Generate a summary of a session's exchanges."""
    conversation = "\n".join([
        f"User: {e.user_message}\nAssistant: {e.assistant_response}"
        for e in exchanges
    ])
    return llm_call(
        "You are a session summariser. Write a concise 2-3 sentence summary of what was discussed, "
        "focusing on the user's personal facts, preferences, and decisions. "
        "CRITICAL: Always refer to the human speaker as 'The user' (e.g., 'The user is named Dr. Elena Vasquez', 'The user works at...'). "
        "Be specific about names, numbers, and details.",
        f"Summarise this conversation:\n\n{conversation}",
        temperature=0.3,
    )


def extract_key_topics(exchanges: list[EpisodicMemory]) -> list[str]:
    """Extract key topics from a session."""
    conversation = "\n".join([e.user_message for e in exchanges])
    result = llm_call(
        "You are a topic extractor. Output ONLY a comma-separated list of 2-5 key topics, nothing else.",
        f"Extract key topics from these messages:\n{conversation}",
        temperature=0.1,
    )
    return [t.strip() for t in result.split(",") if t.strip()][:5]


# ═══════════════════════════════════════════════════════════════════════════
#  PART 6 — MEMORY-AUGMENTED CHATBOT
# ═══════════════════════════════════════════════════════════════════════════

class MemoryChatbot:
    """A chatbot that remembers across sessions using dual memory stores."""

    def __init__(self, user_id: str = "default"):
        self.user_id = user_id
        self.session_id = str(uuid.uuid4())[:8]
        self.episodic_store = EpisodicStore()
        self.semantic_index = SemanticIndex()
        self.retriever = MemoryRetriever(self.episodic_store, self.semantic_index)
        self.current_exchanges: list[EpisodicMemory] = []

    def _build_system_prompt(self, memories: list[RetrievedMemory]) -> str:
        """Build system prompt with injected memories."""
        base_prompt = (
            "You are a helpful, conversational AI assistant with persistent memory. "
            f"You are currently talking to the user with ID '{self.user_id}'. "
            "You remember past conversations with this user across different sessions. "
            "When relevant, naturally reference things you remember from previous conversations. "
            "Be specific about details you recall — names, numbers, preferences, facts. "
            "If you remember something relevant, mention it naturally (e.g., 'I remember you mentioned...'). "
            "Do NOT make up memories — only reference things actually shown in your memory context below. "
            f"IMPORTANT: In the memory context, references to 'The user', 'the user', or '{self.user_id}' ALL mean the person you are currently talking to."
        )

        if memories:
            memory_context = "\n\n--- MEMORIES FROM PAST SESSIONS ---\n"
            for i, mem in enumerate(memories, 1):
                memory_context += f"\nMemory {i} [{mem.source}] (relevance: {mem.blended_score:.2f}):\n"
                memory_context += f"  {mem.content}\n"
            memory_context += "\n--- END OF MEMORIES ---\n"
            memory_context += "\nUse these memories when relevant to the conversation. Be natural about it."
            return base_prompt + memory_context
        else:
            return base_prompt + "\n\nNo past memories found for this user. This appears to be a new conversation."

    def chat(self, user_message: str) -> str:
        """Process a user message with memory augmentation."""
        # 1. Retrieve relevant memories
        memories = self.retriever.retrieve(user_message, self.user_id, top_k=5)

        # 2. Build memory-augmented system prompt
        system_prompt = self._build_system_prompt(memories)

        # 3. Include current session context
        context = ""
        if self.current_exchanges:
            context = "\n\n--- CURRENT SESSION CONTEXT ---\n"
            for ex in self.current_exchanges[-5:]:  # Last 5 exchanges from current session
                context += f"User: {ex.user_message}\nAssistant: {ex.assistant_response}\n\n"

        # 4. Generate response
        full_prompt = context + f"\nUser: {user_message}" if context else user_message
        response = llm_call(system_prompt, full_prompt, temperature=0.7)

        # 5. Score importance and extract topic
        importance = score_importance(user_message, response)
        topic = extract_topic(user_message)

        # 6. Store in episodic memory
        memory = EpisodicMemory(
            session_id=self.session_id,
            user_id=self.user_id,
            user_message=user_message,
            assistant_response=response,
            importance_score=importance,
            topic=topic,
        )
        self.episodic_store.store(memory)
        self.current_exchanges.append(memory)

        return response

    def end_session(self):
        """End the current session — summarise and store in semantic index."""
        if not self.current_exchanges:
            return

        console.print(f"\n[dim]Summarising session {self.session_id[:6]}...[/]")

        # Generate session summary
        summary_text = summarise_session(self.current_exchanges)
        key_topics = extract_key_topics(self.current_exchanges)

        # Store in ChromaDB semantic index
        summary = SessionSummary(
            session_id=self.session_id,
            user_id=self.user_id,
            summary=summary_text,
            key_topics=key_topics,
            num_exchanges=len(self.current_exchanges),
        )
        self.semantic_index.store_summary(summary)

        console.print(f"[green]Session {self.session_id[:6]} saved to memory.[/]")
        console.print(f"[dim]Summary: {summary_text}[/]")
        console.print(f"[dim]Topics: {', '.join(key_topics)}[/]")


# ═══════════════════════════════════════════════════════════════════════════
#  PART 7 — CLI COMMANDS
# ═══════════════════════════════════════════════════════════════════════════

@app.command()
def demo():
    """Run a full 3-session demonstration showing cross-session recall."""
    console.print(Rule("[bold cyan]Lab 6.1 — Memory-Augmented Chatbot Demo[/]"))
    console.print()
    console.print(Panel(
        "[bold]This demo runs 3 automated sessions to demonstrate cross-session recall.[/]\n\n"
        "• Session 1: User shares personal facts (name, job, project)\n"
        "• Session 2: User discusses different topic, agent may recall session 1\n"
        "• Session 3: User asks about something from session 1 — agent must recall it",
        title="[bold cyan]Demo Overview[/]",
        border_style="cyan",
    ))
    console.print()

    # ── Session 1: Establishing facts ──
    console.print(Rule("[bold green]Session 1 — Establishing Personal Facts[/]"))
    bot1 = MemoryChatbot(user_id="demo_user")

    session1_messages = [
        "Hi! My name is Marcus and I work as a machine learning engineer at DeepScale AI.",
        "I'm currently building a recommendation system for e-commerce. We're using a transformer-based architecture with 12 attention heads.",
        "My favourite programming language is Rust, but I use Python daily for ML work. I also enjoy hiking on weekends — my favourite trail is the Blue Ridge Parkway.",
    ]

    for msg in session1_messages:
        console.print(f"\n[bold blue]User:[/] {msg}")
        response = bot1.chat(msg)
        console.print(f"[bold green]Agent:[/] {response}")
        time.sleep(2)  # Rate limit spacing

    bot1.end_session()
    time.sleep(3)

    # ── Session 2: Different topic ──
    console.print(Rule("[bold green]Session 2 — Different Topic (Testing Memory Persistence)[/]"))
    bot2 = MemoryChatbot(user_id="demo_user")

    session2_messages = [
        "I've been reading about knowledge graphs recently. What are the key advantages over traditional databases?",
        "That's interesting. I'm also thinking about adding a graph-based feature to my current project. What graph database would you recommend?",
    ]

    for msg in session2_messages:
        console.print(f"\n[bold blue]User:[/] {msg}")
        response = bot2.chat(msg)
        console.print(f"[bold green]Agent:[/] {response}")
        time.sleep(2)

    bot2.end_session()
    time.sleep(3)

    # ── Session 3: Cross-session recall ──
    console.print(Rule("[bold green]Session 3 — Cross-Session Recall Test[/]"))
    bot3 = MemoryChatbot(user_id="demo_user")

    session3_messages = [
        "Hey, do you remember what company I work at and what project I'm building?",
        "What outdoor activities did I mention I enjoy?",
        "What programming language did I say is my favourite?",
    ]

    for msg in session3_messages:
        console.print(f"\n[bold blue]User:[/] {msg}")
        response = bot3.chat(msg)
        console.print(f"[bold green]Agent:[/] {response}")
        time.sleep(2)

    bot3.end_session()

    # ── Summary ──
    console.print()
    console.print(Rule("[bold cyan]Demo Complete[/]"))
    _render_memory_stats("demo_user")


@app.command()
def session(user: str = typer.Option("alice", help="User ID for this session")):
    """Start an interactive chat session with memory."""
    console.print(Rule(f"[bold cyan]Lab 6.1 — Interactive Session (user: {user})[/]"))
    console.print()

    bot = MemoryChatbot(user_id=user)
    total_memories = bot.episodic_store.count_memories(user)
    sessions = bot.episodic_store.get_all_sessions(user)

    console.print(Panel(
        f"[bold]User:[/] {user}\n"
        f"[bold]Session ID:[/] {bot.session_id}\n"
        f"[bold]Past memories:[/] {total_memories}\n"
        f"[bold]Past sessions:[/] {len(sessions)}\n\n"
        "[dim]Type 'quit' or 'exit' to end the session.[/]",
        title="[bold cyan]Session Info[/]",
        border_style="cyan",
    ))

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not user_input or user_input.lower() in ("quit", "exit"):
            break

        response = bot.chat(user_input)
        console.print(f"[bold green]Agent:[/] {response}")

    bot.end_session()
    console.print(Rule("[bold cyan]Session Ended[/]"))


@app.command()
def validate():
    """Run the 3-session validation test and check cross-session recall."""
    console.print(Rule("[bold cyan]Lab 6.1 — Validation: Cross-Session Recall Test[/]"))
    console.print()

    # Clean up previous validation data
    test_user = f"test_user_{str(uuid.uuid4())[:4]}"
    console.print(f"[dim]Using test user: {test_user}[/]\n")

    results = {"passed": 0, "failed": 0, "details": []}

    # ── Session 1 ──
    console.print(Panel("[bold]Session 1:[/] Establishing facts", border_style="yellow"))
    bot1 = MemoryChatbot(user_id=test_user)

    facts = [
        ("My name is Dr. Elena Vasquez and I'm a neuroscientist at Stanford.", "name", "Elena Vasquez"),
        ("I'm researching the effects of sleep deprivation on synaptic plasticity. My lab has 7 PhD students.", "research", "synaptic plasticity"),
        ("My favourite book is 'The Structure of Scientific Revolutions' by Thomas Kuhn.", "book", "Thomas Kuhn"),
    ]

    for msg, topic, key_fact in facts:
        console.print(f"  [blue]User:[/] {msg}")
        resp = bot1.chat(msg)
        console.print(f"  [green]Agent:[/] {resp[:120]}...")
        time.sleep(2)

    bot1.end_session()
    time.sleep(3)

    # ── Session 2 ──
    console.print(Panel("[bold]Session 2:[/] Different topic (filler)", border_style="yellow"))
    bot2 = MemoryChatbot(user_id=test_user)

    resp = bot2.chat("Tell me about the latest developments in quantum computing.")
    console.print(f"  [blue]User:[/] Tell me about quantum computing developments.")
    console.print(f"  [green]Agent:[/] {resp[:120]}...")
    time.sleep(2)

    bot2.end_session()
    time.sleep(3)

    # ── Session 3: Recall test ──
    console.print(Panel("[bold]Session 3:[/] Cross-session recall test", border_style="yellow"))
    bot3 = MemoryChatbot(user_id=test_user)

    recall_tests = [
        ("What is my name and where do I work?", ["elena", "vasquez", "stanford"], "Name and workplace"),
        ("What am I researching?", ["sleep", "synaptic", "plasticity"], "Research topic"),
        ("What is my favourite book?", ["kuhn", "scientific", "revolutions"], "Favourite book"),
    ]

    for question, expected_keywords, test_name in recall_tests:
        console.print(f"\n  [blue]User:[/] {question}")
        response = bot3.chat(question)
        console.print(f"  [green]Agent:[/] {response}")

        # Check if any expected keywords appear in response
        response_lower = response.lower()
        found = [kw for kw in expected_keywords if kw in response_lower]
        passed = len(found) >= 1

        if passed:
            results["passed"] += 1
            console.print(f"  [green]✓ PASS[/] — {test_name} (found: {', '.join(found)})")
        else:
            results["failed"] += 1
            console.print(f"  [red]✗ FAIL[/] — {test_name} (expected one of: {', '.join(expected_keywords)})")

        results["details"].append({
            "test": test_name,
            "question": question,
            "response": response,
            "expected_keywords": expected_keywords,
            "found_keywords": found,
            "passed": passed,
        })
        time.sleep(2)

    bot3.end_session()

    # ── Results ──
    console.print()
    console.print(Rule("[bold cyan]Validation Results[/]"))

    results_table = Table(title="Cross-Session Recall Results", box=box.ROUNDED)
    results_table.add_column("Test", style="bold")
    results_table.add_column("Result", justify="center")
    results_table.add_column("Keywords Found")

    for detail in results["details"]:
        status = "[green]PASS[/]" if detail["passed"] else "[red]FAIL[/]"
        found = ", ".join(detail["found_keywords"]) if detail["found_keywords"] else "[dim]none[/]"
        results_table.add_row(detail["test"], status, found)

    console.print(results_table)

    total = results["passed"] + results["failed"]
    console.print(f"\n[bold]Score: {results['passed']}/{total}[/]")
    if results["passed"] >= 2:
        console.print("[bold green]✓ Validation PASSED — Agent demonstrates cross-session recall[/]")
    else:
        console.print("[bold red]✗ Validation FAILED — Insufficient cross-session recall[/]")

    # Save report
    report_path = ROOT_DIR / "labs" / "lab_6_1_validation_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    console.print(f"\n[dim]Report saved to {report_path}[/]")


@app.command()
def inspect(user: str = typer.Option("demo_user", help="User ID to inspect")):
    """Inspect the memory stores for a given user."""
    console.print(Rule(f"[bold cyan]Lab 6.1 — Memory Inspector (user: {user})[/]"))
    console.print()
    _render_memory_stats(user)

    # Show recent episodic memories
    store = EpisodicStore()
    memories = store.get_recent(user, limit=10)

    if memories:
        ep_table = Table(title="Recent Episodic Memories", box=box.ROUNDED)
        ep_table.add_column("Session", style="cyan", width=8)
        ep_table.add_column("Timestamp", style="dim", width=20)
        ep_table.add_column("User Message", width=35)
        ep_table.add_column("Topic", style="yellow", width=12)
        ep_table.add_column("Importance", justify="center", width=10)

        for mem in memories:
            imp_color = "green" if mem.importance_score >= 0.7 else "yellow" if mem.importance_score >= 0.4 else "dim"
            ep_table.add_row(
                mem.session_id[:6],
                mem.timestamp[:19],
                mem.user_message[:35] + ("..." if len(mem.user_message) > 35 else ""),
                mem.topic,
                f"[{imp_color}]{mem.importance_score:.2f}[/]",
            )

        console.print(ep_table)
    else:
        console.print("[dim]No episodic memories found.[/]")

    # Show semantic index stats
    console.print()
    sem_index = SemanticIndex()
    console.print(f"[bold]Semantic index entries:[/] {sem_index.count()}")


def _render_memory_stats(user_id: str):
    """Render memory statistics for a user."""
    store = EpisodicStore()
    total = store.count_memories(user_id)
    sessions = store.get_all_sessions(user_id)

    stats_table = Table(title=f"Memory Stats for '{user_id}'", box=box.ROUNDED)
    stats_table.add_column("Metric", style="bold")
    stats_table.add_column("Value", style="cyan")

    stats_table.add_row("Total Episodic Memories", str(total))
    stats_table.add_row("Total Sessions", str(len(sessions)))

    if sessions:
        stats_table.add_row("Session IDs", ", ".join([s[:6] for s in sessions]))

    sem_index = SemanticIndex()
    stats_table.add_row("Semantic Index Entries", str(sem_index.count()))

    console.print(stats_table)


if __name__ == "__main__":
    app()
