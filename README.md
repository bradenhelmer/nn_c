# Neural Network in C

A from-scratch neural network implementation in C, built for learning and understanding the fundamentals of deep learning without high-level frameworks.

## Overview

This project implements a neural network library in C with custom linear algebra operations, activation functions, and training algorithms. Supports both single-layer perceptrons and multi-layer perceptrons (MLPs) with backpropagation.

## Features

### Linear Algebra
- **Matrix Operations**: Creation, multiplication, transpose, addition, subtraction, element-wise operations
- **Vector Operations**: Dot products, element-wise operations, scaling
- **Optimized Memory Management**: Efficient allocation and deallocation

### Activation Functions
- Sigmoid
- ReLU (Rectified Linear Unit)
- Tanh (Hyperbolic Tangent)
- Linear (Identity)
- Softmax

All activation functions include their derivatives for backpropagation.

### Neural Network Components
- **Perceptron**: Single-layer perceptron with configurable activation functions
- **Multi-Layer Perceptron (MLP)**: Fully-connected feedforward neural networks with:
  - Arbitrary number of hidden layers
  - Configurable layer sizes and activations
  - Backpropagation for gradient computation
  - Xavier and He weight initialization
- **Loss Functions**:
  - Mean Squared Error (MSE)
  - Cross-Entropy Loss
  - Softmax Cross-Entropy (numerically stable, for multi-class classification)
- **Training**: Multiple training modes with configurable:
  - **Training Algorithms**:
    - Full-batch gradient descent
    - Mini-batch gradient descent
    - Stochastic gradient descent
  - **Optimizers**:
    - SGD (Stochastic Gradient Descent)
    - Momentum
    - Adam (Adaptive Moment Estimation)
  - **Hyperparameters**:
    - Learning rate with optional scheduling
    - Batch size
    - Maximum epochs
    - Convergence tolerance
    - L2 regularization (weight decay)
  - Verbose output options with per-epoch metrics

### Dataset Handling
- Custom dataset structure with feature matrix and target vector
- Pre-built logic gate datasets (AND, OR, XOR)
- MNIST handwritten digit dataset (60,000 training + 10,000 test images)
- Batched data loading with shuffle support
- Easy extension for custom datasets

## Project Structure

```
nn_c/
├── src/
│   ├── activations/       # Activation functions and derivatives
│   ├── data/             # Dataset creation and management
│   │   ├── dataset.c     # Dataset structure and logic gates
│   │   └── batch.c       # Batch iterator for mini-batch training
│   ├── linalg/           # Linear algebra (matrix and vector operations)
│   ├── nn/               # Neural network components
│   │   ├── perceptron.c  # Single-layer perceptron
│   │   ├── mlp.c         # Multi-layer perceptron
│   │   ├── layer.c       # Dense layer implementation
│   │   └── loss.c        # Loss functions (MSE, Cross-Entropy, Softmax CE)
│   ├── training/         # Training algorithms
│   │   ├── gradient_descent.c  # Full-batch, mini-batch, and SGD
│   │   ├── optimizer.c         # SGD, Momentum, Adam optimizers
│   │   └── scheduler.c         # Learning rate scheduling
│   ├── examples/         # Example applications
│   │   └── mnist_examples.c  # MNIST training demonstrations
│   └── main.c            # Main entry point
├── tests/                # Unit tests for all components
├── data/                 # MNIST data files
├── Makefile              # Build configuration
└── README.md             # This file
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

The default application demonstrates neural network learning through multiple phases:

**Phase 1: Perceptron Learning**
- **AND Gate**: Successfully learns (linearly separable)
- **OR Gate**: Successfully learns (linearly separable)
- **XOR Gate**: Fails to converge (not linearly separable)

The XOR failure demonstrates the fundamental limitation of single-layer perceptrons.

**Phase 2: MLP Learning**
- **XOR Gate**: Successfully learns using a 2-2-1 architecture (2 inputs, 2 hidden neurons, 1 output)

This demonstrates how multi-layer networks can learn non-linearly separable functions.

**Phase 4: MNIST Classification**
- **MNIST Dataset**: Handwritten digit classification (784 inputs, 128 hidden, 10 outputs)
- Multiple training demonstrations with different optimizers:
  - SGD (Stochastic Gradient Descent)
  - Momentum
  - Adam
- Achieves 90%+ accuracy on MNIST test set
- Includes mini-batch training with configurable batch sizes

## Testing

The project includes comprehensive unit tests for all major components:

```bash
make test
```

Tests cover:
- Vector operations
- Matrix operations
- Activation functions
- Perceptron forward and backward passes
- Layer forward and backward passes
- MLP gradient computation
- Loss function derivatives
- Optimizer implementations (SGD, Momentum, Adam)
- Batch iterator functionality

## Example Output

```
=== Perceptron Learning Logic Gates ===

Training AND Gate...
AND gate training completed in 4523 epochs
Final loss: 0.000001
Final accuracy: 100.000000

Training OR Gate...
OR gate training completed in 3891 epochs
Final loss: 0.000001
Final accuracy: 100.000000

Training XOR Gate (should fail to converge)...
XOR Gate Training stopped at 100 epochs
Final loss: 0.250156 (should be high)
Final accuracy: 0.00% (should be ~0%)

Note: XOR is not linearly separable - single perceptron cannot learn it!
This motivates Phase 3: Multi-layer perceptrons.


Training XOR Gate with MLP...

XOR Gate Training stopped at 8234 epochs
Final loss: 0.000012
Final accuracy: 100.00%

Testing MLP on XOR Gate:
Input: [0.00, 0.00] -> Output: 0.02 (Expected: 0.00) ✓
Input: [0.00, 1.00] -> Output: 0.98 (Expected: 1.00) ✓
Input: [1.00, 0.00] -> Output: 0.97 (Expected: 1.00) ✓
Input: [1.00, 1.00] -> Output: 0.03 (Expected: 0.00) ✓
```

## Development Phases

### Phase 1: Foundation (Complete)
- Linear algebra operations (vectors and matrices)
- Basic activation functions
- Memory management

### Phase 2: Perceptron (Complete)
- Single-layer perceptron implementation
- Gradient descent training
- Loss functions
- Dataset handling
- Logic gate demonstrations

### Phase 3: Multi-Layer Perceptron (Complete)
- Dense layer implementation with forward and backward passes
- Backpropagation algorithm
- Xavier and He weight initialization
- Multiple loss functions (MSE, Cross-Entropy)
- Configurable network architectures
- Successfully learns XOR and other non-linearly separable functions

### Phase 4: Advanced Training & MNIST (Complete)
- **Mini-Batch Training**:
  - Batch iterator with shuffling support
  - Configurable batch sizes
  - Full-batch, mini-batch, and stochastic gradient descent
- **Optimizers**:
  - SGD (Stochastic Gradient Descent)
  - Momentum optimizer with velocity tracking
  - Adam optimizer with adaptive learning rates
  - Direct array access for performance optimization
- **MNIST Dataset**:
  - Batched MNIST file loader (60,000 train + 10,000 test)
  - Softmax cross-entropy loss for multi-class classification
  - Achieves 90%+ test accuracy
- **Additional Features**:
  - L2 regularization (weight decay)
  - Learning rate scheduling
  - Memory-optimized training loops (pre-allocated vectors)
  - Per-epoch loss and accuracy tracking

### Future Enhancements
- Additional optimizers (RMSprop, AdaGrad, Nadam)
- Regularization techniques (L1, dropout, batch normalization)
- More activation functions (Leaky ReLU, ELU, Swish, GELU)
- Convolutional layers
- More complex datasets (CIFAR-10, Fashion-MNIST)
- Model saving and loading
- Early stopping and checkpointing
- Data augmentation

## Code Style

The project follows a consistent C coding style:
- Snake case for functions and variables
- Descriptive naming conventions
- Clear separation of concerns
- Comprehensive comments
- Formatted with clang-format

## Performance Optimizations

The codebase includes several performance optimizations:
- **Direct Array Access**: Optimizers use direct array indexing instead of function calls for matrix/vector operations
- **Memory Reuse**: Training loops pre-allocate vectors and reuse them across iterations
- **Single-Pass Operations**: Combined operations (e.g., `W -= lr * dW`) reduce multiple loops to one
- **Numerically Stable Implementations**: Softmax uses max subtraction to prevent overflow

## Memory Management

All dynamically allocated structures include corresponding `_free()` functions. The project is designed to be leak-free (verify with `make memcheck`).

## License

This is an educational project. Feel free to use and modify as needed.

## Contributing

This is a personal learning project, but suggestions and improvements are welcome.

## Acknowledgments

Built from scratch to understand neural networks at a fundamental level, without relying on high-level machine learning frameworks.
