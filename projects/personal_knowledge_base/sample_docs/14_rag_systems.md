# RAG Systems (Retrieval-Augmented Generation)

RAG combines information retrieval with language generation to produce grounded, accurate responses.

## Why RAG?
- LLMs have knowledge cutoffs and can hallucinate
- RAG grounds responses in actual documents
- Enables working with private, domain-specific data
- More cost-effective than fine-tuning for knowledge updates

## RAG Architecture

### Ingestion Pipeline
1. Document Loading: PDF, HTML, DOCX, etc.
2. Text Splitting: Chunk documents into manageable pieces
3. Embedding: Convert chunks to vectors using an embedding model
4. Storage: Store vectors and metadata in a vector database

### Query Pipeline
1. Query Embedding: Convert user question to a vector
2. Retrieval: Find top-k most similar document chunks
3. Context Assembly: Format retrieved chunks as context
4. Generation: LLM generates answer using context + question
5. Citation: Include source references in the response

## Advanced RAG Patterns
- **Hybrid Search**: Combine keyword (BM25) and semantic search
- **Re-ranking**: Cross-encoder to re-order retrieved results
- **Multi-query**: Generate query variations for broader retrieval
- **HyDE**: Generate hypothetical documents to improve retrieval
- **Parent-Document Retrieval**: Retrieve small chunks, return parent documents
- **Self-RAG**: Model decides when to retrieve and evaluates its own output