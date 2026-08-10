"""
Enterprise AI Memory Platform — Semantic Memory Store
======================================================
ChromaDB-backed semantic memory with per-tenant collection
namespacing and similarity search.
"""

import uuid
from pathlib import Path
from typing import Optional

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from models import SemanticMemory, SemanticMemoryCreate, SemanticQueryRequest

import chromadb


class SemanticStore:
    """Per-tenant semantic memory store backed by ChromaDB."""

    def __init__(self, persist_dir: str = "data/chromadb"):
        self.persist_dir = persist_dir
        Path(persist_dir).parent.mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self._collections: dict[str, chromadb.Collection] = {}

    def _get_collection(self, tenant_id: str) -> chromadb.Collection:
        """Get or create a tenant-namespaced collection."""
        key = f"semantic_{tenant_id}"
        if key not in self._collections:
            self._collections[key] = self.client.get_or_create_collection(
                name=key,
                metadata={"hnsw:space": "cosine"},
            )
        return self._collections[key]

    def store(self, tenant_id: str, memory: SemanticMemoryCreate) -> SemanticMemory:
        """Store a semantic fact/preference for a tenant."""
        entry = SemanticMemory(
            memory_id=str(uuid.uuid4())[:12],
            tenant_id=tenant_id,
            fact=memory.fact,
            category=memory.category,
            confidence=memory.confidence,
            metadata=memory.metadata,
        )

        collection = self._get_collection(tenant_id)
        meta = {
            "category": entry.category,
            "confidence": entry.confidence,
            "timestamp": entry.timestamp,
            "tenant_id": tenant_id,
        }
        # Merge user-provided metadata (stringify non-primitives)
        for k, v in entry.metadata.items():
            if isinstance(v, (str, int, float, bool)):
                meta[k] = v
            else:
                meta[k] = str(v)

        collection.upsert(
            ids=[entry.memory_id],
            documents=[entry.fact],
            metadatas=[meta],
        )
        return entry

    def query(self, tenant_id: str, request: SemanticQueryRequest) -> list[SemanticMemory]:
        """Query semantic memories by similarity."""
        collection = self._get_collection(tenant_id)
        if collection.count() == 0:
            return []

        where_filter = None
        if request.category:
            where_filter = {"category": request.category}

        results = collection.query(
            query_texts=[request.query],
            n_results=min(request.limit, collection.count()),
            where=where_filter if where_filter else None,
        )

        memories = []
        if results and results["documents"] and results["documents"][0]:
            for i, doc in enumerate(results["documents"][0]):
                meta = results["metadatas"][0][i] if results["metadatas"] else {}
                mid = results["ids"][0][i] if results["ids"] else str(uuid.uuid4())[:12]
                distance = results["distances"][0][i] if results["distances"] else 1.0
                relevance = max(0.0, 1.0 - distance)

                memories.append(SemanticMemory(
                    memory_id=mid,
                    tenant_id=tenant_id,
                    timestamp=meta.get("timestamp", ""),
                    fact=doc,
                    category=meta.get("category", "general"),
                    confidence=float(meta.get("confidence", 0.8)),
                    metadata={"relevance_score": round(relevance, 4)},
                ))
        return memories

    def delete(self, tenant_id: str, memory_id: str) -> bool:
        """Delete a semantic memory by ID."""
        collection = self._get_collection(tenant_id)
        try:
            collection.delete(ids=[memory_id])
            return True
        except Exception:
            return False

    def count(self, tenant_id: str) -> int:
        """Count semantic memories for a tenant."""
        collection = self._get_collection(tenant_id)
        return collection.count()

    def get_all(self, tenant_id: str, limit: int = 100) -> list[SemanticMemory]:
        """Get all semantic memories for a tenant."""
        collection = self._get_collection(tenant_id)
        if collection.count() == 0:
            return []

        results = collection.get(limit=min(limit, collection.count()))
        memories = []
        if results and results["documents"]:
            for i, doc in enumerate(results["documents"]):
                meta = results["metadatas"][i] if results["metadatas"] else {}
                mid = results["ids"][i] if results["ids"] else str(uuid.uuid4())[:12]
                memories.append(SemanticMemory(
                    memory_id=mid,
                    tenant_id=tenant_id,
                    timestamp=meta.get("timestamp", ""),
                    fact=doc,
                    category=meta.get("category", "general"),
                    confidence=float(meta.get("confidence", 0.8)),
                    metadata={k: v for k, v in meta.items() if k not in ("category", "confidence", "timestamp", "tenant_id")},
                ))
        return memories
