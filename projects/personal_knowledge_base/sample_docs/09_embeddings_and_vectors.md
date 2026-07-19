# Embeddings and Vector Spaces

Embeddings are dense vector representations of data that capture semantic relationships.

## What Are Embeddings?
- A mapping from high-dimensional, sparse data (text, images) to lower-dimensional, dense vectors
- Similar items are mapped to nearby points in the vector space
- Typical dimensions: 384 (MiniLM), 768 (BERT), 1536 (OpenAI text-embedding-3)

## Text Embeddings
### Word Embeddings
- **Word2Vec**: Skip-gram and CBOW architectures (Mikolov et al., 2013)
- **GloVe**: Global Vectors, captures co-occurrence statistics
- **FastText**: Handles subword information, works with out-of-vocabulary words

### Sentence Embeddings
- **Sentence-BERT**: Fine-tuned BERT for sentence-level similarity
- **all-MiniLM-L6-v2**: Fast, lightweight model (384 dimensions)
- **BGE**: High-quality bilingual embeddings from BAAI

## Similarity Measures
- **Cosine Similarity**: Measures angle between vectors (most common)
- **Euclidean Distance**: Straight-line distance between points
- **Dot Product**: Product of magnitudes and cosine of angle
- **Manhattan Distance**: Sum of absolute differences

## Applications
- Semantic search and information retrieval
- Recommendation systems
- Clustering and classification
- Anomaly detection
- RAG (Retrieval-Augmented Generation)