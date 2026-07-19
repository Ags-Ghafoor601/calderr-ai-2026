# Generative AI

Generative AI creates new content — text, images, audio, code — that resembles human-created output.

## Key Technologies

### Large Language Models (LLMs)
Trained on vast text corpora to predict the next token.
Examples: GPT-4, Claude, Gemini, LLaMA, Mistral.
Applications: chatbots, content creation, code generation, translation.

### Diffusion Models
Generate images by gradually denoising random noise.
Process: Forward diffusion (add noise) → Reverse diffusion (remove noise to generate image).
Examples: Stable Diffusion, DALL-E 3, Midjourney.

### Generative Adversarial Networks (GANs)
Two networks compete: Generator creates fake data, Discriminator identifies fakes.
Through adversarial training, the generator learns to create realistic outputs.
Applications: image synthesis, style transfer, data augmentation.

### Variational Autoencoders (VAEs)
Learn a compressed latent representation of data.
Can generate new data by sampling from the latent space.
Used in: drug discovery, anomaly detection, image generation.

## Challenges
- Hallucination: LLMs can generate plausible but incorrect information
- Bias: Models inherit biases present in training data
- Copyright: Generated content may resemble copyrighted material
- Environmental cost: Training large models requires significant energy