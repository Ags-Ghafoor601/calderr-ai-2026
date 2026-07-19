# Neural Networks Explained

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