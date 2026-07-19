# The Transformer Architecture

The Transformer (Vaswani et al., 2017) is the foundation of modern AI.

## Architecture Overview
The original Transformer has an encoder-decoder structure:

### Encoder
- Stack of N identical layers (N=6 in the original paper)
- Each layer has: Multi-Head Self-Attention → Add & Norm → Feed-Forward → Add & Norm
- Processes the input sequence in parallel

### Decoder
- Stack of N identical layers
- Each layer has: Masked Multi-Head Self-Attention → Cross-Attention → Feed-Forward
- Generates output auto-regressively (one token at a time)

## Key Components
1. **Positional Encoding**: Sinusoidal functions to inject position information
2. **Multi-Head Attention**: 8 parallel attention heads in the original paper
3. **Feed-Forward Network**: Two linear transformations with ReLU activation
4. **Layer Normalization**: Stabilizes training by normalizing activations
5. **Residual Connections**: Skip connections to enable gradient flow

## Scaling Laws
- Performance improves predictably with more parameters, data, and compute
- Chinchilla scaling: Optimal ratio of ~20 tokens per parameter
- Emergent abilities appear at certain scale thresholds

## Variants
- **GPT**: Decoder-only, autoregressive
- **BERT**: Encoder-only, bidirectional
- **T5**: Full encoder-decoder
- **Vision Transformer (ViT)**: Applies transformers to image patches