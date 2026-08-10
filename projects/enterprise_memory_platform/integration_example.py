"""
Enterprise AI Memory Platform — LangChain Agent Integration Example
=====================================================================
Demonstrates how an external LangChain agent connects to the
Memory Platform via REST API to:
  1. Store episodic memories after each interaction
  2. Retrieve relevant semantic facts before responding
  3. Apply procedural correction rules
  4. Build knowledge graph entries from conversations

This is a standalone example — it connects to the FastAPI service
via HTTP, proving the platform works as a true microservice.

Usage:
    1. Start the API:  uvicorn api:app --port 8000
    2. Run this:       python integration_example.py
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Optional

import requests
from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

API_BASE = os.getenv("MEMORY_API_URL", "http://localhost:8000")
TENANT_ID = "langchain_demo_agent"
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")


class MemoryClient:
    """HTTP client for the Enterprise AI Memory Platform API."""

    def __init__(self, base_url: str = API_BASE, tenant_id: str = TENANT_ID):
        self.base_url = base_url.rstrip("/")
        self.tenant_id = tenant_id
        self.session = requests.Session()
        self.session.headers["Content-Type"] = "application/json"
        self._ensure_tenant()

    def _ensure_tenant(self):
        """Create tenant if it doesn't exist."""
        try:
            self.session.post(
                f"{self.base_url}/tenants",
                params={"tenant_id": self.tenant_id, "name": "LangChain Demo Agent"},
            )
        except requests.ConnectionError:
            print(f"WARNING: Could not connect to Memory Platform at {self.base_url}")
            print("Make sure the API is running: uvicorn api:app --port 8000")

    # ── Episodic ──────────────────────────────────────────────────

    def store_episode(self, session_id: str, content: str, role: str = "user",
                      importance: float = 0.5) -> dict:
        resp = self.session.post(
            f"{self.base_url}/memory/{self.tenant_id}/episodic",
            json={"session_id": session_id, "content": content,
                  "role": role, "importance_score": importance},
        )
        return resp.json()

    def recall_episodes(self, session_id: Optional[str] = None, limit: int = 10) -> list[dict]:
        resp = self.session.post(
            f"{self.base_url}/memory/{self.tenant_id}/episodic/query",
            json={"session_id": session_id, "limit": limit},
        )
        data = resp.json()
        return data.get("data", [])

    # ── Semantic ──────────────────────────────────────────────────

    def store_fact(self, fact: str, category: str = "general",
                   confidence: float = 0.8) -> dict:
        resp = self.session.post(
            f"{self.base_url}/memory/{self.tenant_id}/semantic",
            json={"fact": fact, "category": category, "confidence": confidence},
        )
        return resp.json()

    def search_facts(self, query: str, limit: int = 5) -> list[dict]:
        resp = self.session.post(
            f"{self.base_url}/memory/{self.tenant_id}/semantic/query",
            json={"query": query, "limit": limit},
        )
        data = resp.json()
        return data.get("data", [])

    # ── Procedural ────────────────────────────────────────────────

    def store_rule(self, mistake: str, correction: str, rule: str,
                   domain: str = "general", confidence: float = 0.7) -> dict:
        resp = self.session.post(
            f"{self.base_url}/memory/{self.tenant_id}/procedural",
            json={
                "original_mistake": mistake, "correction": correction,
                "rule_text": rule, "domain": domain, "confidence": confidence,
            },
        )
        return resp.json()

    def get_rules(self, domain: Optional[str] = None, limit: int = 10) -> list[dict]:
        body: dict = {"limit": limit, "active_only": True}
        if domain:
            body["domain"] = domain
        resp = self.session.post(
            f"{self.base_url}/memory/{self.tenant_id}/procedural/query",
            json=body,
        )
        data = resp.json()
        return data.get("data", [])

    # ── Knowledge Graph ───────────────────────────────────────────

    def add_entity(self, name: str, entity_type: str = "concept",
                   description: str = "") -> dict:
        resp = self.session.post(
            f"{self.base_url}/memory/{self.tenant_id}/graph/entity",
            json={"name": name, "entity_type": entity_type, "description": description},
        )
        return resp.json()

    def add_relationship(self, source: str, target: str,
                         relation_type: str, confidence: float = 0.8) -> dict:
        resp = self.session.post(
            f"{self.base_url}/memory/{self.tenant_id}/graph/relationship",
            json={"source": source, "target": target,
                  "relation_type": relation_type, "confidence": confidence},
        )
        return resp.json()

    # ── Stats ─────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        resp = self.session.get(f"{self.base_url}/tenants/{self.tenant_id}/stats")
        return resp.json()


class MemoryAugmentedAgent:
    """A simple LLM agent that uses the Memory Platform for all 4 memory types.

    This demonstrates the integration pattern:
    1. Before responding → retrieve relevant memories
    2. After responding → store the interaction as episodic memory
    3. On correction → extract a procedural rule
    4. Periodically → extract knowledge graph entries
    """

    def __init__(self, memory_client: MemoryClient, session_id: str = "demo-session"):
        self.memory = memory_client
        self.session_id = session_id
        self.interaction_count = 0

    def respond(self, user_message: str) -> str:
        """Generate a response augmented with memories from the platform."""
        self.interaction_count += 1

        # 1. Store user message as episodic
        self.memory.store_episode(
            self.session_id, user_message, role="user", importance=0.5,
        )

        # 2. Retrieve relevant context
        relevant_facts = self.memory.search_facts(user_message, limit=3)
        recent_episodes = self.memory.recall_episodes(self.session_id, limit=5)
        rules = self.memory.get_rules(limit=5)

        # 3. Build augmented prompt
        context_parts = []
        if relevant_facts:
            facts_text = "\n".join(f"- {f['fact']}" for f in relevant_facts)
            context_parts.append(f"Known facts:\n{facts_text}")
        if rules:
            rules_text = "\n".join(f"- [{r['domain']}] {r['rule_text']}" for r in rules)
            context_parts.append(f"Rules to follow:\n{rules_text}")
        if recent_episodes:
            history = "\n".join(
                f"[{e['role']}]: {e['content'][:100]}" for e in recent_episodes[-5:]
            )
            context_parts.append(f"Recent conversation:\n{history}")

        context = "\n\n".join(context_parts) if context_parts else "No prior context."

        # 4. Generate response via Groq
        response = self._llm_call(
            "You are a helpful AI assistant with persistent memory. "
            "Use the provided context to give personalised, consistent responses. "
            "Follow all rules from past corrections.",
            f"Context:\n{context}\n\nUser: {user_message}",
        )

        # 5. Store assistant response as episodic
        self.memory.store_episode(
            self.session_id, response, role="assistant", importance=0.5,
        )

        return response

    def learn_from_correction(self, original_question: str, original_answer: str,
                              correction: str):
        """Extract and store a procedural rule from a correction."""
        rule_text = self._llm_call(
            "Extract a generalised rule from this correction. "
            "The rule should be general enough to apply to similar future situations. "
            "Return ONLY the rule text, nothing else.",
            f"Question: {original_question}\n"
            f"Wrong answer: {original_answer}\n"
            f"Correction: {correction}",
        )

        self.memory.store_rule(
            mistake=original_answer[:200],
            correction=correction[:200],
            rule=rule_text[:200],
            domain="general",
            confidence=0.8,
        )
        print(f"  [Learned rule] {rule_text[:100]}")

    def extract_knowledge(self, text: str):
        """Extract entities and relationships from text and store in the graph."""
        entities_json = self._llm_call(
            'Extract entities from this text. Return a JSON list of objects with '
            '"name" and "type" fields. Types: person, company, concept, place, product. '
            'Return ONLY valid JSON.',
            text,
        )

        import re
        try:
            cleaned = entities_json.strip()
            if cleaned.startswith("```"):
                cleaned = re.sub(r"```(?:json)?\s*", "", cleaned)
                cleaned = cleaned.rstrip("`").strip()
            entities = json.loads(cleaned)
            for ent in entities[:10]:
                self.memory.add_entity(
                    name=ent.get("name", ""),
                    entity_type=ent.get("type", "concept"),
                )
                print(f"  [Entity] {ent.get('name')}: {ent.get('type')}")
        except (json.JSONDecodeError, TypeError):
            print("  [Warning] Could not parse entities")

    def _llm_call(self, system: str, user: str) -> str:
        """Make an LLM call via Groq."""
        try:
            from groq import Groq
            client = Groq(api_key=GROQ_API_KEY)
            for attempt in range(3):
                try:
                    resp = client.chat.completions.create(
                        model="llama-3.1-8b-instant",
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": user},
                        ],
                        temperature=0.7,
                        max_tokens=512,
                    )
                    return resp.choices[0].message.content.strip()
                except Exception as e:
                    if "429" in str(e):
                        time.sleep((attempt + 1) * 10)
                    else:
                        raise
        except ImportError:
            return "LLM not available — install groq package."
        return "Unable to generate response."


# ═══════════════════════════════════════════════════════════════════════════
#  DEMO SCRIPT
# ═══════════════════════════════════════════════════════════════════════════

def run_demo():
    """Run the full integration demo."""
    print("=" * 70)
    print("  Enterprise AI Memory Platform — LangChain Integration Demo")
    print("=" * 70)

    # Initialise
    client = MemoryClient()
    agent = MemoryAugmentedAgent(client)

    # 1. Store some facts
    print("\n--- Seeding Semantic Memory ---")
    facts = [
        ("The user's name is Alice", "profile", 0.95),
        ("Alice prefers Python over JavaScript", "preference", 0.9),
        ("Alice is working on a machine learning project", "knowledge", 0.85),
        ("Alice's project deadline is December 2025", "fact", 0.8),
    ]
    for fact, cat, conf in facts:
        client.store_fact(fact, cat, conf)
        print(f"  Stored: {fact}")

    # 2. Add knowledge graph entries
    print("\n--- Building Knowledge Graph ---")
    client.add_entity("Alice", "person", "The primary user")
    client.add_entity("ML Project", "concept", "Alice's machine learning project")
    client.add_entity("Python", "technology", "Alice's preferred language")
    client.add_relationship("Alice", "ML Project", "working_on")
    client.add_relationship("Alice", "Python", "prefers")
    client.add_relationship("ML Project", "Python", "built_with")
    print("  Added entities and relationships")

    # 3. Interact
    print("\n--- Agent Interactions ---")
    questions = [
        "Hi, can you recommend a good ML library for my project?",
        "What language should I use for data preprocessing?",
        "When is my project deadline?",
    ]

    for q in questions:
        print(f"\n  User: {q}")
        response = agent.respond(q)
        print(f"  Agent: {response[:200]}")
        time.sleep(2)

    # 4. Correction
    print("\n--- Learning from Correction ---")
    agent.learn_from_correction(
        "What language should I use?",
        "I recommend JavaScript for your ML project.",
        "I prefer Python, not JavaScript. Always suggest Python for my ML work.",
    )
    time.sleep(2)

    # 5. Show stats
    print("\n--- Tenant Statistics ---")
    stats = client.get_stats()
    print(f"  Episodic memories:  {stats.get('episodic_count', 0)}")
    print(f"  Semantic memories:  {stats.get('semantic_count', 0)}")
    print(f"  Procedural rules:  {stats.get('procedural_count', 0)}")
    print(f"  Graph entities:    {stats.get('graph_entities', 0)}")
    print(f"  Graph relations:   {stats.get('graph_relationships', 0)}")
    print(f"  Total memories:    {stats.get('total_memories', 0)}")

    print("\n" + "=" * 70)
    print("  Integration demo complete!")
    print("=" * 70)


if __name__ == "__main__":
    run_demo()
