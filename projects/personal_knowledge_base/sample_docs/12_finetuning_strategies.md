# Fine-tuning Strategies

Fine-tuning adapts pre-trained models to specific tasks or domains.

## Full Fine-tuning
Update all model parameters on the target dataset.
- Pros: Maximum expressiveness, best performance
- Cons: Expensive, requires significant GPU memory, risk of catastrophic forgetting

## Parameter-Efficient Fine-tuning (PEFT)

### LoRA (Low-Rank Adaptation)
Inject trainable low-rank matrices into transformer layers.
Only trains 0.1-1% of total parameters.
Original weights remain frozen.
Highly effective for adapting LLMs to specific domains.

### QLoRA
Combines LoRA with 4-bit quantization.
Enables fine-tuning of large models (65B+) on a single GPU.

### Prefix Tuning
Prepend trainable continuous vectors to the input.
The model learns task-specific prefixes while keeping weights frozen.

### Adapter Layers
Insert small trainable modules between transformer layers.
Each adapter has a down-projection, activation, and up-projection.

## Best Practices
1. Start with a small learning rate (1e-5 to 5e-5)
2. Use warmup steps (5-10% of total steps)
3. Monitor validation loss closely
4. Use early stopping to prevent overfitting
5. Evaluate on held-out test set for final metrics