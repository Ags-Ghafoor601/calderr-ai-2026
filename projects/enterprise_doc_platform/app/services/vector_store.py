"""Enterprise Document Intelligence Platform — Multi-tenant Vector Store Service.

Manages ChromaDB collections with tenant-level namespace isolation.
Each tenant gets their own collection to ensure complete data isolation.
"""

import logging
from typing import Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from app.config import settings

logger = logging.getLogger(__name__)


class VectorStoreService:
    """Multi-tenant vector store using ChromaDB with namespace isolation."""

    def __init__(self):
        self._client = chromadb.PersistentClient(path=settings.chroma_dir)
        self._embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )
        logger.info("VectorStoreService initialized (ChromaDB at %s)", settings.chroma_dir)

    # ---- Collection Management ----
    def _collection_name(self, tenant_id: str) -> str:
        """Generate a namespaced collection name for a tenant."""
        return f"tenant_{tenant_id}_docs"

    def create_tenant_collection(self, tenant_id: str) -> chromadb.Collection:
        """Create a new collection for a tenant."""
        name = self._collection_name(tenant_id)
        try:
            self._client.delete_collection(name)
        except Exception:
            pass
        collection = self._client.create_collection(
            name=name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine", "tenant_id": tenant_id},
        )
        logger.info("Created collection '%s' for tenant '%s'", name, tenant_id)
        return collection

    def get_tenant_collection(self, tenant_id: str) -> chromadb.Collection:
        """Get an existing tenant collection."""
        name = self._collection_name(tenant_id)
        return self._client.get_collection(name=name, embedding_function=self._embedding_fn)

    def get_or_create_collection(self, tenant_id: str) -> chromadb.Collection:
        """Get or create a tenant collection."""
        name = self._collection_name(tenant_id)
        return self._client.get_or_create_collection(
            name=name,
            embedding_function=self._embedding_fn,
            metadata={"hnsw:space": "cosine", "tenant_id": tenant_id},
        )

    def delete_tenant_collection(self, tenant_id: str):
        """Delete a tenant's collection (and all their data)."""
        name = self._collection_name(tenant_id)
        try:
            self._client.delete_collection(name)
            logger.info("Deleted collection '%s'", name)
        except Exception as e:
            logger.warning("Failed to delete collection '%s': %s", name, e)

    def tenant_exists(self, tenant_id: str) -> bool:
        """Check if a tenant collection exists."""
        name = self._collection_name(tenant_id)
        try:
            self._client.get_collection(name=name, embedding_function=self._embedding_fn)
            return True
        except Exception:
            return False

    # ---- Document Storage ----
    def add_chunks(
        self,
        tenant_id: str,
        chunks: list[dict],
        document_id: str,
    ) -> int:
        """Add document chunks to a tenant's collection.

        Args:
            tenant_id: Tenant identifier
            chunks: List of {"text": str, "metadata": dict}
            document_id: Unique document identifier

        Returns:
            Number of chunks added.
        """
        collection = self.get_or_create_collection(tenant_id)

        ids = []
        documents = []
        metadatas = []

        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_chunk_{i}"
            ids.append(chunk_id)
            documents.append(chunk["text"])
            metadatas.append({
                **chunk["metadata"],
                "document_id": document_id,
                "tenant_id": tenant_id,
            })

        # Batch add
        batch_size = 500
        for start in range(0, len(ids), batch_size):
            end = start + batch_size
            collection.add(
                ids=ids[start:end],
                documents=documents[start:end],
                metadatas=metadatas[start:end],
            )

        logger.info(
            "Added %d chunks for doc '%s' in tenant '%s'",
            len(chunks), document_id, tenant_id,
        )
        return len(chunks)

    # ---- Retrieval ----
    def semantic_search(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 5,
        document_filter: Optional[str] = None,
    ) -> list[dict]:
        """Search a tenant's documents using semantic similarity.

        Args:
            tenant_id: Tenant identifier
            query: Search query text
            top_k: Number of results to return
            document_filter: Optional document_id to filter by

        Returns:
            List of result dicts with text, metadata, and distance.
        """
        collection = self.get_tenant_collection(tenant_id)

        where_filter = None
        if document_filter:
            where_filter = {"document_id": document_filter}

        results = collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_filter,
        )

        retrieved = []
        for i in range(len(results["documents"][0])):
            retrieved.append({
                "text": results["documents"][0][i],
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i] if results.get("distances") else 0,
                "score": 1 - results["distances"][0][i] if results.get("distances") else 0.5,
            })

        return retrieved

    # ---- Stats ----
    def get_tenant_stats(self, tenant_id: str) -> dict:
        """Get statistics for a tenant's collection."""
        try:
            collection = self.get_tenant_collection(tenant_id)
            all_data = collection.get(include=["metadatas"])
            doc_ids = set()
            for meta in all_data["metadatas"]:
                doc_ids.add(meta.get("document_id", "unknown"))

            return {
                "chunk_count": collection.count(),
                "document_count": len(doc_ids),
                "documents": list(doc_ids),
            }
        except Exception:
            return {"chunk_count": 0, "document_count": 0, "documents": []}

    def get_all_tenant_ids(self) -> list[str]:
        """Get all tenant IDs from existing collections."""
        collections = self._client.list_collections()
        tenant_ids = []
        for col in collections:
            name = col.name if hasattr(col, "name") else str(col)
            if name.startswith("tenant_") and name.endswith("_docs"):
                tenant_id = name[len("tenant_"):-len("_docs")]
                tenant_ids.append(tenant_id)
        return tenant_ids

    def get_all_chunks_for_tenant(self, tenant_id: str) -> list[dict]:
        """Get all chunks for a tenant (used by BM25 indexing)."""
        collection = self.get_tenant_collection(tenant_id)
        all_data = collection.get(include=["documents", "metadatas"])

        chunks = []
        for i in range(len(all_data["documents"])):
            chunks.append({
                "text": all_data["documents"][i],
                "metadata": all_data["metadatas"][i],
            })
        return chunks


# Module-level singleton
vector_store = VectorStoreService()
