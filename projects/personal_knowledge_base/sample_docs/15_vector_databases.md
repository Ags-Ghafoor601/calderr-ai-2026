# Vector Databases

Vector databases are specialized systems for storing, indexing, and querying high-dimensional vectors.

## Why Vector Databases?
Traditional databases use exact matching. Vector databases use approximate nearest neighbor (ANN)
search to find semantically similar items efficiently.

## Popular Options

### ChromaDB
- Open-source, embedded vector database
- Perfect for prototyping and small-to-medium applications
- Python-native API, simple setup
- Supports persistent and in-memory modes

### FAISS (Facebook AI Similarity Search)
- High-performance library for similarity search
- Supports GPU acceleration
- Index types: Flat (exact), IVF (inverted file), HNSW
- Best for: Large-scale, latency-critical applications

### Qdrant
- Production-ready vector database with REST API
- Supports filtering, payload storage, and multi-tenancy
- Cloud and self-hosted options

### Pinecone
- Fully managed vector database service
- Serverless architecture, automatic scaling
- Enterprise features: namespaces, metadata filtering

## Index Types
- **Flat**: Exact search, O(n) complexity. Best for small datasets.
- **IVF (Inverted File)**: Partitions space into clusters. Searches only relevant clusters.
- **HNSW (Hierarchical Navigable Small World)**: Graph-based. Fast and accurate.
- **PQ (Product Quantization)**: Compresses vectors for memory efficiency.