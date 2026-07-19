# Transfer Learning

Transfer learning leverages knowledge gained from one task to improve performance on a different but related task.

## Why Transfer Learning?
- Reduces training time and computational cost
- Works well with limited labeled data
- Pre-trained models capture general features applicable to many tasks

## Approaches

### Feature Extraction
Use a pre-trained model as a fixed feature extractor.
Remove the final classification layer and add new layers for the target task.
Freeze the pre-trained weights during training.

### Fine-tuning
Unfreeze some or all layers of the pre-trained model.
Train with a lower learning rate to adapt the model to the new task.
Typically unfreeze later layers first (they capture task-specific features).

## Popular Pre-trained Models
- **Vision**: ResNet, VGG, EfficientNet, ViT (Vision Transformer)
- **NLP**: BERT, GPT, RoBERTa, T5, LLaMA
- **Multimodal**: CLIP, DALL-E, Flamingo

## Best Practices
1. Start with feature extraction; fine-tune if performance is insufficient
2. Use data augmentation to prevent overfitting
3. Monitor validation loss to detect overfitting early
4. Use learning rate schedulers (warmup + decay)