"""
Enterprise AI Memory Platform — Consolidation Worker
======================================================
Async background worker that:
  1. Summarises old episodic memories into compressed blocks
  2. Promotes high-usage procedural rules
  3. Prunes low-importance consolidated memories
  4. Consolidates similar procedural rules

Runs on a configurable schedule (default: every 100 episodes per tenant).
"""

import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional

from dotenv import load_dotenv

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import ConsolidationRecord, ConsolidationStatus, ConsolidationConfig
from router import MemoryRouter

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
load_dotenv(ROOT_DIR / ".env")

logger = logging.getLogger("consolidation_worker")


def _llm_summarise(episodes: list[dict]) -> str:
    """Summarise a batch of episodic memories using Groq LLM."""
    try:
        from groq import Groq
        api_key = os.getenv("GROQ_API_KEY", "")
        if not api_key:
            return _fallback_summarise(episodes)

        client = Groq(api_key=api_key)
        model = "llama-3.1-8b-instant"

        content_lines = []
        for ep in episodes:
            content_lines.append(f"[{ep.get('role', 'user')}]: {ep.get('content', '')[:200]}")

        prompt = "\n".join(content_lines)

        for attempt in range(3):
            try:
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content":
                            "Summarise the following conversation exchanges into 2-3 sentences. "
                            "Capture the key topics discussed, any decisions made, and important "
                            "facts mentioned. Be concise and factual."},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0.2,
                    max_tokens=256,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                if "429" in str(e):
                    time.sleep((attempt + 1) * 10)
                else:
                    break

    except ImportError:
        pass

    return _fallback_summarise(episodes)


def _fallback_summarise(episodes: list[dict]) -> str:
    """Fallback summarisation without LLM — extract key sentences."""
    if not episodes:
        return "No episodes to summarise."

    key_sentences = []
    for ep in episodes[:10]:
        content = ep.get("content", "")
        sentences = content.split(". ")
        if sentences:
            key_sentences.append(sentences[0][:100])

    return "Summary: " + ". ".join(key_sentences[:5]) + "."


class ConsolidationWorker:
    """Background worker for memory consolidation and maintenance."""

    def __init__(self, router: MemoryRouter, config: Optional[ConsolidationConfig] = None):
        self.router = router
        self.config = config or ConsolidationConfig()
        self.records: list[ConsolidationRecord] = []

    def run_for_tenant(self, tenant_id: str) -> ConsolidationRecord:
        """Run consolidation for a single tenant."""
        record = ConsolidationRecord(
            tenant_id=tenant_id,
            status=ConsolidationStatus.RUNNING,
        )

        try:
            # 1. Check if consolidation is needed
            unconsolidated = self.router.episodic.count_unconsolidated(tenant_id)
            if unconsolidated < self.config.episode_threshold:
                record.status = ConsolidationStatus.COMPLETED
                record.summary_text = f"No consolidation needed ({unconsolidated} episodes, threshold {self.config.episode_threshold})"
                self.records.append(record)
                return record

            # 2. Get oldest unconsolidated episodes
            episodes = self.router.episodic.get_oldest_unconsolidated(
                tenant_id, limit=self.config.batch_size
            )

            if not episodes:
                record.status = ConsolidationStatus.COMPLETED
                record.summary_text = "No episodes to consolidate"
                self.records.append(record)
                return record

            record.episodes_processed = len(episodes)

            # 3. Group episodes by session
            session_groups: dict[str, list] = {}
            for ep in episodes:
                session_groups.setdefault(ep.session_id, []).append(ep)

            # 4. Summarise each session group
            consolidated_count = 0
            for session_id, session_episodes in session_groups.items():
                # Only consolidate low-importance episodes
                to_consolidate = [
                    ep for ep in session_episodes
                    if ep.importance_score < 0.8  # Keep high-importance episodes
                ]

                if to_consolidate:
                    ep_dicts = [{"role": ep.role, "content": ep.content} for ep in to_consolidate]
                    summary = _llm_summarise(ep_dicts)

                    # Store summary as a high-importance semantic memory
                    from models import SemanticMemoryCreate
                    self.router.store_semantic(
                        tenant_id,
                        SemanticMemoryCreate(
                            fact=f"Session {session_id} summary: {summary}",
                            category="session_summary",
                            confidence=0.9,
                        ),
                    )

                    # Mark episodes as consolidated
                    ids_to_mark = [ep.memory_id for ep in to_consolidate]
                    self.router.episodic.mark_consolidated(ids_to_mark)
                    consolidated_count += len(ids_to_mark)

            record.episodes_consolidated = consolidated_count

            # 5. Prune low-importance consolidated memories
            pruned = self.router.episodic.prune_low_importance(
                tenant_id, threshold=self.config.min_importance_to_keep * 0.5
            )
            record.memories_pruned = pruned

            # 6. Promote high-usage procedural rules
            promoted = self.router.procedural.promote_rules(tenant_id)
            record.rules_promoted = promoted

            # 7. Consolidate similar procedural rules
            self.router.procedural.consolidate(tenant_id)

            record.status = ConsolidationStatus.COMPLETED
            record.summary_text = (
                f"Consolidated {consolidated_count}/{len(episodes)} episodes, "
                f"pruned {pruned} memories, promoted {promoted} rules"
            )

        except Exception as e:
            record.status = ConsolidationStatus.FAILED
            record.summary_text = f"Error: {str(e)}"
            logger.error(f"Consolidation failed for tenant {tenant_id}: {e}")

        self.records.append(record)
        return record

    def run_all_tenants(self) -> list[ConsolidationRecord]:
        """Run consolidation for all active tenants."""
        tenants = self.router.tenants.list_tenants(active_only=True)
        results = []
        for tenant in tenants:
            record = self.run_for_tenant(tenant.tenant_id)
            results.append(record)
        return results

    def force_consolidation(self, tenant_id: str, batch_size: Optional[int] = None) -> ConsolidationRecord:
        """Force consolidation regardless of threshold."""
        original = self.config.episode_threshold
        self.config.episode_threshold = 0  # Force trigger
        if batch_size:
            self.config.batch_size = batch_size

        result = self.run_for_tenant(tenant_id)

        self.config.episode_threshold = original
        return result

    def get_history(self, tenant_id: Optional[str] = None) -> list[ConsolidationRecord]:
        """Get consolidation history."""
        if tenant_id:
            return [r for r in self.records if r.tenant_id == tenant_id]
        return self.records
