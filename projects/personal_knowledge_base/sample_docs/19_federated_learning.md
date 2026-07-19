# Federated Learning

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