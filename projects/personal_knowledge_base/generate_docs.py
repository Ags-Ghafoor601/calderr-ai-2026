#!/usr/bin/env python3
"""Generate 22 sample knowledge base documents for the Personal Knowledge Base project."""
import os

DOCS_DIR = os.path.join(os.path.dirname(__file__), "sample_docs")
os.makedirs(DOCS_DIR, exist_ok=True)

DOCUMENTS = {
    "01_machine_learning_basics.md": """# Introduction to Machine Learning

Machine learning (ML) is a subset of artificial intelligence that focuses on building systems that learn from data.
Instead of being explicitly programmed, ML algorithms identify patterns in data and make decisions with minimal human intervention.

## Types of Machine Learning

### Supervised Learning
In supervised learning, models are trained on labeled datasets. Each training example consists of an input-output pair.
Common algorithms include linear regression, decision trees, random forests, and neural networks.
Applications: spam detection, image classification, price prediction.

### Unsupervised Learning
Unsupervised learning works with unlabeled data. The algorithm tries to find hidden patterns or groupings.
Key techniques include k-means clustering, hierarchical clustering, PCA, and autoencoders.
Applications: customer segmentation, anomaly detection, dimensionality reduction.

### Reinforcement Learning
In reinforcement learning, an agent learns by interacting with an environment and receiving rewards or penalties.
The agent learns a policy that maximizes cumulative reward over time.
Applications: game playing (AlphaGo), robotics, autonomous driving.

## The ML Pipeline
1. Data Collection → 2. Data Preprocessing → 3. Feature Engineering → 4. Model Selection →
5. Training → 6. Evaluation → 7. Hyperparameter Tuning → 8. Deployment → 9. Monitoring
""",

    "02_neural_networks.md": """# Neural Networks Explained

Neural networks are computing systems inspired by the biological neural networks in the human brain.
They consist of interconnected nodes (neurons) organized in layers.

## Architecture

### Input Layer
Receives the raw input features. Each neuron represents one feature of the input data.

### Hidden Layers
Process information through weighted connections. Deep networks have multiple hidden layers.
Each neuron applies a weighted sum followed by an activation function (ReLU, sigmoid, tanh).

### Output Layer
Produces the final prediction. For classification, often uses softmax activation.

## Training Process
Neural networks learn through backpropagation and gradient descent:
1. Forward pass: Input flows through the network to produce output
2. Loss calculation: Compare output with expected result
3. Backward pass: Compute gradients of loss with respect to weights
4. Weight update: Adjust weights in the direction that reduces loss

## Key Concepts
- **Learning Rate**: Controls the step size during weight updates
- **Batch Size**: Number of samples processed before updating weights
- **Epochs**: Number of complete passes through the training dataset
- **Overfitting**: Model memorizes training data instead of learning patterns
- **Dropout**: Regularization technique that randomly disables neurons during training
""",

    "03_natural_language_processing.md": """# Natural Language Processing (NLP)

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
""",

    "04_computer_vision.md": """# Computer Vision Overview

Computer vision enables machines to interpret and understand visual information from the world.

## Key Applications

### Image Classification
Categorizing images into predefined classes. CNNs (Convolutional Neural Networks) are the backbone.
Notable architectures: AlexNet, VGG, ResNet, EfficientNet.

### Object Detection
Locating and classifying objects within images.
Key models: YOLO (You Only Look Once), Faster R-CNN, SSD.

### Semantic Segmentation
Classifying each pixel in an image into a category.
Used in autonomous driving, medical imaging, and satellite analysis.

### Image Generation
Creating new images from noise or text descriptions.
Technologies: GANs (Generative Adversarial Networks), Diffusion Models, DALL-E, Stable Diffusion.

## Convolutional Neural Networks (CNNs)
- Convolutional layers: Learn spatial features using filters
- Pooling layers: Reduce spatial dimensions (max pooling, average pooling)
- Fully connected layers: Final classification based on extracted features
- Transfer learning: Using pre-trained models (ImageNet) as starting points
""",

    "05_reinforcement_learning.md": """# Reinforcement Learning

Reinforcement learning (RL) is a paradigm where agents learn optimal behavior through trial and error.

## Key Concepts
- **Agent**: The learner or decision-maker
- **Environment**: The world the agent interacts with
- **State**: Current situation of the agent
- **Action**: What the agent can do
- **Reward**: Feedback signal indicating how good an action was
- **Policy**: Strategy that the agent follows (maps states to actions)

## Algorithms

### Q-Learning
A model-free algorithm that learns the value of state-action pairs.
The Q-table stores expected cumulative rewards for each state-action combination.

### Deep Q-Networks (DQN)
Combines Q-learning with deep neural networks to handle large state spaces.
Used by DeepMind to achieve superhuman performance in Atari games.

### Policy Gradient Methods
Directly optimize the policy function rather than the value function.
REINFORCE algorithm uses the policy gradient theorem.

### Actor-Critic Methods
Combine value-based and policy-based approaches.
A2C, A3C, and PPO (Proximal Policy Optimization) are popular variants.

## Applications
- Game AI (AlphaGo, OpenAI Five)
- Robotics (manipulation, locomotion)
- Resource management and scheduling
- Recommendation systems
""",

    "06_transfer_learning.md": """# Transfer Learning

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
""",

    "07_generative_ai.md": """# Generative AI

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
""",

    "08_large_language_models.md": """# Large Language Models (LLMs)

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
""",

    "09_embeddings_and_vectors.md": """# Embeddings and Vector Spaces

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
""",

    "10_attention_mechanism.md": """# The Attention Mechanism

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
""",

    "11_transformer_architecture.md": """# The Transformer Architecture

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
""",

    "12_finetuning_strategies.md": """# Fine-tuning Strategies

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
""",

    "13_prompt_engineering.md": """# Prompt Engineering

Prompt engineering is the art of crafting effective inputs to get desired outputs from LLMs.

## Core Techniques

### Zero-Shot Prompting
Ask the model directly without any examples.
"Classify the sentiment of this review: 'The movie was fantastic!'"

### Few-Shot Prompting
Provide a few examples before the actual task.
This helps the model understand the expected format and reasoning pattern.

### Chain-of-Thought (CoT)
Instruct the model to think step-by-step.
"Let's think step by step" dramatically improves reasoning accuracy.

### ReAct (Reasoning + Acting)
Combine reasoning traces with actions.
The model thinks about what to do, takes an action, observes the result, and repeats.

## Advanced Techniques
- **Self-Consistency**: Generate multiple answers and take the majority vote
- **Tree of Thoughts**: Explore multiple reasoning paths in a tree structure
- **Prompt Chaining**: Break complex tasks into a series of simpler prompts
- **System Prompts**: Set the model's role, tone, and constraints

## Anti-Patterns to Avoid
- Vague or ambiguous instructions
- Missing context or constraints
- Not specifying output format
- Ignoring the model's limitations
- Prompt injection vulnerabilities
""",

    "14_rag_systems.md": """# RAG Systems (Retrieval-Augmented Generation)

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
""",

    "15_vector_databases.md": """# Vector Databases

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
""",

    "16_ai_ethics.md": """# AI Ethics and Responsible AI

As AI systems become more powerful, ethical considerations become increasingly critical.

## Key Ethical Principles

### Fairness
AI systems should treat all groups equitably.
- Bias in training data leads to biased predictions
- Disparate impact: Different outcomes for different demographic groups
- Mitigation: Diverse training data, fairness metrics, regular audits

### Transparency
Users should understand how AI decisions are made.
- Explainable AI (XAI): Techniques to make model decisions interpretable
- Model cards: Documentation of model capabilities and limitations
- Algorithmic auditing: Regular assessment of model behavior

### Privacy
Protecting individual data rights.
- Differential privacy: Adding noise to prevent individual identification
- Federated learning: Training without centralizing data
- Data minimization: Collecting only necessary data

### Accountability
Clear responsibility for AI system outcomes.
- Human-in-the-loop: Human oversight for critical decisions
- Incident response: Plans for when AI systems fail
- Regulatory compliance: GDPR, AI Act, industry standards

## Emerging Challenges
- Deepfakes and synthetic media
- Autonomous weapons and military AI
- Job displacement and economic impact
- Environmental cost of training large models
- AI alignment: Ensuring AI goals align with human values
""",

    "17_ai_in_healthcare.md": """# AI in Healthcare

AI is transforming healthcare through improved diagnostics, drug discovery, and patient care.

## Applications

### Medical Imaging
- Radiology: AI detects tumors, fractures, and abnormalities in X-rays, CT, MRI
- Pathology: Analyzing tissue samples at cellular level
- Dermatology: Skin cancer detection from photographs
- Accuracy often matches or exceeds human specialists

### Drug Discovery
- Molecular property prediction using graph neural networks
- Virtual screening of millions of drug candidates
- AlphaFold: Predicting protein structures for drug target identification
- Reduces drug development timeline from years to months

### Clinical Decision Support
- Predicting patient deterioration in ICU settings
- Recommending treatment plans based on patient history
- Identifying drug interactions and contraindications
- Risk stratification for chronic diseases

### Natural Language Processing in Healthcare
- Clinical note summarization
- Medical coding and billing automation
- Literature review and evidence synthesis
- Patient communication chatbots

## Challenges
- Data privacy (HIPAA compliance)
- Regulatory approval (FDA, CE marking)
- Clinical validation and safety testing
- Integration with existing healthcare IT systems
- Addressing health disparities and bias
""",

    "18_ai_in_finance.md": """# AI in Finance

The financial industry leverages AI for trading, risk management, fraud detection, and customer service.

## Key Applications

### Algorithmic Trading
- High-frequency trading using ML models
- Sentiment analysis of news and social media
- Portfolio optimization using reinforcement learning
- Predictive models for asset prices and market trends

### Fraud Detection
- Real-time transaction monitoring
- Anomaly detection using autoencoders
- Graph neural networks for detecting fraud rings
- Behavioral biometrics for identity verification

### Credit Risk Assessment
- Alternative data sources for credit scoring
- ML models outperform traditional scorecards
- Explainable AI requirements for lending decisions
- Fair lending compliance and bias detection

### Customer Service
- AI chatbots for banking queries
- Personalized financial advice
- Document processing (KYC, loan applications)
- Voice-based authentication

## Risk Management
- Stress testing portfolios with ML simulations
- Early warning systems for market crashes
- Regulatory compliance monitoring
- Cybersecurity threat detection

## Regulatory Considerations
- Model interpretability requirements
- Fair lending laws (ECOA, Fair Housing Act)
- Data protection regulations (GDPR, CCPA)
- Model risk management guidelines (SR 11-7)
""",

    "19_federated_learning.md": """# Federated Learning

Federated learning enables training ML models across decentralized data sources without sharing raw data.

## How It Works
1. Central server sends model to participating devices/organizations
2. Each participant trains the model on their local data
3. Only model updates (gradients) are sent back to the server
4. Server aggregates updates to improve the global model
5. Process repeats for multiple rounds

## Types

### Cross-Device
- Training across millions of mobile devices
- Examples: Google Gboard keyboard predictions, Apple Siri
- Challenges: device heterogeneity, communication constraints

### Cross-Silo
- Training across organizations (hospitals, banks)
- Fewer participants but larger datasets
- Examples: multi-hospital medical research

## Key Techniques
- **FedAvg**: Federated Averaging — the foundational algorithm
- **Differential Privacy**: Add noise to prevent data leakage
- **Secure Aggregation**: Encrypt model updates during transmission
- **Model Compression**: Reduce communication overhead

## Advantages
- Data never leaves the device/organization
- Complies with privacy regulations (GDPR)
- Leverages diverse data sources
- Reduces data centralization risks

## Challenges
- Non-IID data distribution across participants
- Communication efficiency
- Handling stragglers and dropouts
- Ensuring model convergence
""",

    "20_mlops.md": """# MLOps: Machine Learning Operations

MLOps applies DevOps principles to machine learning systems for reliable, scalable deployment.

## ML Lifecycle
1. Data Management → 2. Experimentation → 3. Training → 4. Evaluation →
5. Deployment → 6. Monitoring → 7. Retraining

## Key Practices

### Experiment Tracking
- Log hyperparameters, metrics, and artifacts
- Tools: MLflow, Weights & Biases, Neptune
- Reproducibility: version code, data, and environment

### Model Registry
- Centralized storage for trained models
- Version control for models
- Stage management: staging → production → archived

### CI/CD for ML
- Automated training pipelines
- Model validation gates (accuracy thresholds)
- Automated deployment with rollback capability

### Model Monitoring
- Data drift detection: Input distribution changes
- Model drift: Performance degradation over time
- Prediction monitoring: Track inference latency and errors
- Alerting: Automated notifications when metrics degrade

## Infrastructure
- **Containerization**: Docker for reproducible environments
- **Orchestration**: Kubernetes for scaling
- **Feature Stores**: Centralized feature management (Feast, Tecton)
- **Data Versioning**: DVC (Data Version Control)

## Best Practices
- Automate everything possible
- Test data quality, not just code quality
- Monitor models in production continuously
- Document model decisions and limitations
""",

    "21_automl.md": """# AutoML: Automated Machine Learning

AutoML automates the end-to-end process of applying machine learning to real-world problems.

## What AutoML Automates

### Feature Engineering
- Automatic feature selection and creation
- Handling missing values and outliers
- Encoding categorical variables
- Feature scaling and normalization

### Model Selection
- Evaluating multiple algorithms automatically
- Comparing performance across model families
- Ensemble methods: combining multiple models

### Hyperparameter Optimization
- **Grid Search**: Exhaustive search over parameter grid
- **Random Search**: Random sampling of parameter space
- **Bayesian Optimization**: Intelligent search using surrogate models
- **Neural Architecture Search (NAS)**: Designing neural network architectures

## Popular Tools
- **Auto-sklearn**: Automated scikit-learn pipelines
- **H2O AutoML**: Enterprise-grade automated ML
- **Google AutoML**: Cloud-based automated model training
- **TPOT**: Genetic programming-based pipeline optimization
- **AutoGluon**: Amazon's AutoML toolkit

## Limitations
- May not handle domain-specific requirements
- Black-box nature reduces interpretability
- Computational cost of searching large spaces
- Still requires human expertise for problem framing
""",

    "22_data_engineering.md": """# Data Engineering for AI

Data engineering builds the infrastructure and pipelines that feed ML systems with clean, reliable data.

## Data Pipeline Architecture

### Batch Processing
- Process large volumes of data on a schedule
- Tools: Apache Spark, Apache Beam, dbt
- Use cases: Training data preparation, feature computation

### Stream Processing
- Process data in real-time as it arrives
- Tools: Apache Kafka, Apache Flink, Apache Storm
- Use cases: Real-time predictions, fraud detection

### ETL vs ELT
- **ETL**: Extract, Transform, Load (transform before storage)
- **ELT**: Extract, Load, Transform (transform after storage)
- Modern data stacks favor ELT with powerful cloud warehouses

## Data Quality
- **Completeness**: No missing critical values
- **Accuracy**: Data correctly represents reality
- **Consistency**: Same data across different systems
- **Timeliness**: Data is up-to-date
- **Validity**: Data conforms to expected formats

## Storage Solutions
- **Data Lakes**: Raw, unstructured data (S3, GCS, ADLS)
- **Data Warehouses**: Structured, optimized for analytics (Snowflake, BigQuery)
- **Feature Stores**: ML-specific feature management (Feast, Tecton)
- **Vector Databases**: Embedding storage for AI (ChromaDB, Pinecone)

## Best Practices
- Schema versioning and evolution
- Data lineage tracking
- Automated data quality checks
- Idempotent pipeline design
- Documentation and data catalogs
""",
}

def main():
    for filename, content in DOCUMENTS.items():
        filepath = os.path.join(DOCS_DIR, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content.strip())
    print(f"Created {len(DOCUMENTS)} sample documents in {DOCS_DIR}")

if __name__ == "__main__":
    main()
