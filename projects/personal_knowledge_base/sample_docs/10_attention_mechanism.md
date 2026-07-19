# The Attention Mechanism

Attention allows neural networks to focus on relevant parts of the input when producing output.

## Self-Attention
Each element in the sequence attends to all other elements to compute a weighted representation.

### Query, Key, Value
- **Query (Q)**: What am I looking for?
- **Key (K)**: What do I contain?
- **Value (V)**: What information do I provide?
- Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) × V

### Multi-Head Attention
Run multiple attention operations in parallel with different learned projections.
Each head can attend to different types of relationships.
Outputs are concatenated and linearly projected.

## Types of Attention
- **Self-attention**: Sequence attends to itself
- **Cross-attention**: One sequence attends to another (e.g., decoder to encoder)
- **Causal attention**: Each position can only attend to previous positions (used in GPT)

## Key Properties
- Captures long-range dependencies without recurrence
- Parallelizable computation (unlike RNNs)
- Quadratic complexity O(n²) with sequence length
- Flash Attention: Optimized implementation reducing memory usage

## Impact
The attention mechanism enabled the Transformer revolution, leading to breakthroughs in NLP, computer vision, and multimodal AI.