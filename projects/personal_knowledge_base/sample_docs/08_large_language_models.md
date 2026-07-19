# Large Language Models (LLMs)

LLMs are neural networks trained on massive text datasets to understand and generate human language.

## Architecture
Most modern LLMs are based on the Transformer architecture:
- **Decoder-only** (GPT family): Autoregressive, generates text left-to-right
- **Encoder-only** (BERT family): Bidirectional, good for understanding tasks
- **Encoder-decoder** (T5, BART): Best for sequence-to-sequence tasks

## Training Pipeline
1. **Pre-training**: Self-supervised learning on trillions of tokens
2. **Supervised Fine-tuning (SFT)**: Training on curated instruction-response pairs
3. **RLHF**: Reinforcement Learning from Human Feedback for alignment

## Key Capabilities
- Text generation and completion
- Question answering and summarization
- Code generation and debugging
- Reasoning and chain-of-thought
- Tool use and function calling
- Multi-turn conversation

## Inference Optimization
- **Quantization**: Reducing precision (FP32 → INT8/INT4) to reduce memory
- **KV Cache**: Caching key-value pairs to avoid recomputation
- **Speculative Decoding**: Using a smaller model to draft tokens
- **Batching**: Processing multiple requests simultaneously