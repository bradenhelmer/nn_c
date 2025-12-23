# Neural Network in C

A from-scratch neural network implementation in C, built for learning and understanding the fundamentals of deep learning without high-level frameworks.

## Overview

This project implements a neural network library in C with a **tensor-based architecture**, custom linear algebra operations, activation functions, and training algorithms. The framework supports multiple layer types through a generic layer interface, enabling both simple perceptrons and deep multi-layer networks with backpropagation.

## Features

### Tensor Operations
- **Multi-dimensional Arrays**: 1D to 4D tensor support with flexible shape management
- **Core Operations**: Element-wise operations, matrix-vector multiplication, outer products, transposed multiplication
- **Memory Efficient**: Pre-computed strides for fast indexing, zero-copy row extraction
- **Utilities**: Clone, flatten, unflatten, padding/unpadding for convolutions

### Activation Functions
- Sigmoid (with derivative: s(1-s) computed from output)
- ReLU (Rectified Linear Unit)
- Tanh (Hyperbolic Tangent)
- Linear (Identity)
- Softmax

All activation functions include tensor-based derivatives for backpropagation.

### Generic Layer System
The framework uses a generic `Layer` interface that wraps type-specific implementations:

- **Linear Layer**: Fully-connected dense layer (`output = Wx + b`)
  - Xavier and He weight initialization
  - Gradient accumulation for batch training
- **Activation Layer**: Applies activation functions with cached input/output for backprop
- **Convolutional Layer** (Conv2D): 2D convolutions with configurable filters, stride, and padding
- **Max Pooling Layer**: Spatial downsampling with index tracking for backprop

### Neural Network Components
- **Perceptron**: Single-layer perceptron with configurable activation functions
- **Multi-Layer Networks**: Arbitrary depth with mixed layer types
  - Chain-based forward pass
  - Automatic backpropagation through layer stack
  - Per-layer gradient zeroing and weight updates
- **Loss Functions**:
  - Mean Squared Error (MSE)
  - Cross-Entropy Loss
  - Softmax Cross-Entropy (numerically stable, for multi-class classification)

### Training Infrastructure
- **Training Algorithms**:
  - Stochastic Gradient Descent (per-sample updates)
  - Mini-batch Gradient Descent (accumulated gradients)
  - Full-batch Gradient Descent
- **Optimizers**:
  - SGD (Stochastic Gradient Descent)
  - Momentum (velocity tracking)
  - Adam (Adaptive Moment Estimation)
- **Learning Rate Scheduling**:
  - Step decay
  - Exponential decay
  - Cosine annealing
- **Regularization**:
  - L2 regularization (weight decay)
- **Training Features**:
  - Configurable batch sizes with shuffle support
  - Early stopping based on convergence tolerance
  - Per-epoch loss and accuracy tracking
  - Verbose output options

### Dataset Handling
- Custom dataset structure with feature tensor and target tensor
- Pre-built logic gate datasets (AND, OR, XOR)
- MNIST handwritten digit dataset (60,000 training + 10,000 test images)
- Batch iterator with shuffle support
- Easy extension for custom datasets

## Project Structure

```
nn_c/
├── src/
│   ├── tensor/              # Tensor data structure and operations
│   │   └── tensor.c         # Multi-dimensional array implementation
│   ├── activations/         # Activation functions (scalar and tensor)
│   │   └── activations.c    # Sigmoid, ReLU, Tanh, Softmax with derivatives
│   ├── data/                # Dataset creation and management
│   │   ├── dataset.c        # Dataset structure and logic gates
│   │   └── batch.c          # Batch iterator for mini-batch training
│   ├── nn/                  # Neural network components
│   │   ├── layer.c          # Generic layer interface (dispatch to specific types)
│   │   ├── linear_layer.c   # Dense fully-connected layer
│   │   ├── activation_layer.c # Activation function wrapper layer
│   │   ├── conv_layer.c     # 2D convolutional layer
│   │   ├── pool_layer.c     # Max pooling layer
│   │   ├── nn.c             # Neural network (layer composition, forward/backward)
│   │   ├── perceptron.c     # Single-layer perceptron
│   │   └── loss.c           # Loss functions (MSE, Cross-Entropy, Softmax CE)
│   ├── training/            # Training algorithms
│   │   ├── gradient_descent.c  # SGD, mini-batch, and batch training
│   │   ├── optimizer.c         # SGD, Momentum, Adam optimizers
│   │   └── scheduler.c         # Learning rate scheduling
│   ├── examples/            # Example applications
│   │   ├── perceptron_examples.c  # Logic gate demonstrations
│   │   ├── nn_examples.c          # XOR with MLP
│   │   └── mnist_examples.c       # MNIST training demonstrations
│   ├── utils/               # Utility functions
│   └── main.c               # Main entry point
├── tests/                   # Unit tests for all components
├── data/                    # MNIST data files
├── Makefile                 # Build configuration
└── README.md                # This file
```

## Building

### Prerequisites
- C compiler (clang or gcc)
- Make
- Standard C math library

### Build Commands

```bash
# Build the main executable and tests
make

# Build and run the main program
make run

# Run unit tests
make test

# Build with debug symbols
make debug

# Format code with clang-format
make format

# Clean build artifacts
make clean

# Memory leak check (requires valgrind)
make memcheck
```

The compiled binary will be in `build/bin/neural_net`.

## Running

```bash
# After building
./build/bin/neural_net
```

The application demonstrates neural network learning through multiple phases:

### Phase 1: Perceptron Learning
- **AND Gate**: Successfully learns (linearly separable)
- **OR Gate**: Successfully learns (linearly separable)
- **XOR Gate**: Fails to converge (not linearly separable)

The XOR failure demonstrates the fundamental limitation of single-layer perceptrons.

### Phase 2: MLP Learning
- **XOR Gate**: Successfully learns using a 2-2-1 architecture
  - 2 inputs → 2 hidden neurons (sigmoid) → 1 output (sigmoid)
  - Demonstrates non-linear function learning

### Phase 3: MNIST Classification
Handwritten digit classification with a 784-128-10 architecture:
- **784 inputs** (28x28 pixel images flattened)
- **128 hidden neurons** (ReLU activation)
- **10 outputs** (softmax for digit classes 0-9)

Multiple optimizer demonstrations achieving **97-99% test accuracy**:

| Optimizer | Training Accuracy | Test Accuracy | Notes |
|-----------|-------------------|---------------|-------|
| SGD | 98.33% | 97.74% | Baseline |
| Momentum | 98.33% | 97.45% | Velocity tracking |
| Adam | 99.38% | 97.93% | Adaptive learning rates |
| Adam + Cosine + L2 | 99.20% | 97.85% | With scheduling and regularization |

## Example Output

```
Training MNIST with SGD optimizer...
Epoch 0: loss=0.4166, accuracy=88.73%
Epoch 1: loss=0.2180, accuracy=93.81%
Epoch 2: loss=0.1635, accuracy=95.44%
...
Epoch 9: loss=0.0593, accuracy=98.33%

MNIST Batched Training with SGD Optimizer stopped at 10 epochs
Final loss: 0.059259
Final accuracy: 98.33%

Testing MNIST test images on batched SGD-trained 2-layer NN:
Final images correctly classified: 9774


Training MNIST with ADAM optimizer...
Epoch 0: loss=0.3261, accuracy=91.14%
Epoch 1: loss=0.1525, accuracy=95.58%
Epoch 2: loss=0.1054, accuracy=96.85%
...
Epoch 9: loss=0.0222, accuracy=99.38%

MNIST Batched Training with ADAM optimizer stopped at 10 epochs
Final loss: 0.022198
Final accuracy: 99.38%

Testing MNIST test images on batched ADAM-trained 2-layer NN:
Final images correctly classified: 9793
```

## Testing

The project includes comprehensive unit tests for all major components:

```bash
make test
```

Tests cover:
- Tensor operations (creation, indexing, arithmetic)
- Activation functions and derivatives
- Perceptron forward and backward passes
- Layer forward and backward passes (Linear, Activation, Conv, Pool)
- Neural network gradient computation
- Loss function derivatives
- Optimizer implementations (SGD, Momentum, Adam)
- Batch iterator functionality

## Architecture

### Tensor-Based Design
The framework uses tensors as the fundamental data type, enabling:
- Unified interface for 1D vectors, 2D matrices, and higher-dimensional data
- Efficient memory layout with pre-computed strides
- Easy extension to convolutional networks (3D/4D tensors for images)

### Generic Layer Interface
```c
typedef struct {
    LayerType type;  // LAYER_LINEAR, LAYER_ACTIVATION, LAYER_CONV_2D, LAYER_MAX_POOL
    void *layer;     // Pointer to specific layer implementation
} Layer;
```

This design allows:
- Polymorphic layer handling in the neural network
- Easy addition of new layer types
- Clean separation between interface and implementation

### Backpropagation Flow
```
Forward:  Input → Linear → Activation → Linear → Activation → Output
                    ↓           ↓          ↓          ↓
                (cache)     (cache)    (cache)    (cache)

Backward: Loss Gradient → Activation' → Linear' → Activation' → Linear' → Input Gradient
                              ↓            ↓           ↓           ↓
                          (accumulate gradients for weight updates)
```

## Development Phases

### Phase 1: Foundation (Complete)
- Tensor data structure with multi-dimensional support
- Basic activation functions with derivatives
- Memory management

### Phase 2: Perceptron (Complete)
- Single-layer perceptron implementation
- Gradient descent training
- Loss functions
- Dataset handling
- Logic gate demonstrations

### Phase 3: Generic Layer Framework (Complete)
- Abstract `Layer` interface with type dispatch
- Linear layer with forward/backward passes
- Activation layer wrapping activation functions
- Xavier and He weight initialization
- Successfully learns XOR and other non-linearly separable functions

### Phase 4: Advanced Training & MNIST (Complete)
- **Mini-Batch Training**: Batch iterator with shuffling, gradient accumulation
- **Optimizers**: SGD, Momentum, Adam with direct tensor operations
- **MNIST Dataset**: 60K train + 10K test, achieving 97-99% accuracy
- **Learning Rate Scheduling**: Step, exponential, cosine annealing
- **Regularization**: L2 weight decay

### Phase 5: Convolutional Layers (In Progress)
- Conv2D layer with configurable filters, stride, padding
- Max pooling layer with index tracking
- Support for 3D tensors (channels × height × width)

### Future Enhancements
- Additional optimizers (RMSprop, AdaGrad, Nadam)
- More regularization (L1, dropout, batch normalization)
- More activations (Leaky ReLU, ELU, Swish, GELU)
- CNN demonstrations (MNIST with convolutions)
- Model saving and loading
- Early stopping and checkpointing

## Performance Optimizations

- **Direct Array Access**: Optimizers use direct indexing for tensor operations
- **Memory Reuse**: Training loops pre-allocate tensors and reuse across iterations
- **Cached Computations**: Activation layers cache output for efficient derivative computation
- **Numerically Stable**: Softmax uses max subtraction to prevent overflow

## Memory Management

All dynamically allocated structures include corresponding `_free()` functions. The project is designed to be leak-free (verify with `make memcheck`).

## License

This is an educational project. Feel free to use and modify as needed.

## Contributing

This is a personal learning project, but suggestions and improvements are welcome.

## Acknowledgments

Built from scratch to understand neural networks at a fundamental level, without relying on high-level machine learning frameworks.
