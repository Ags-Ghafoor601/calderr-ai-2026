# Natural Language Processing (NLP)

NLP is the branch of AI that helps computers understand, interpret, and generate human language.

## Core NLP Tasks

### Text Classification
Assigning predefined categories to text documents.
Examples: sentiment analysis, topic classification, spam detection.

### Named Entity Recognition (NER)
Identifying and classifying named entities (persons, organizations, locations) in text.

### Machine Translation
Automatically translating text from one language to another.
Modern approaches use encoder-decoder architectures with attention mechanisms.

### Text Summarization
Producing concise summaries of longer documents.
- Extractive: Selecting key sentences from the original text
- Abstractive: Generating new sentences that capture the main ideas

### Question Answering
Building systems that can answer questions posed in natural language.
RAG (Retrieval-Augmented Generation) combines retrieval with generative models.

## The Transformer Revolution
The transformer architecture (Vaswani et al., 2017) revolutionized NLP:
- Self-attention mechanism allows processing all tokens in parallel
- Positional encoding captures word order without recurrence
- Models like BERT, GPT, and T5 are all based on transformers