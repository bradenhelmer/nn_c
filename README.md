# Neural Network in C

A from-scratch neural network implementation in C, built for learning and understanding the fundamentals of deep learning without high-level frameworks.

## Overview

This project implements a neural network library in C with custom linear algebra operations, activation functions, and training algorithms. Currently supports single-layer perceptrons with plans for multi-layer architectures.

## Features

### Linear Algebra
- **Matrix Operations**: Creation, multiplication, transpose, addition, subtraction, element-wise operations
- **Vector Operations**: Dot products, element-wise operations, scaling
- **Optimized Memory Management**: Efficient allocation and deallocation

### Activation Functions
- Sigmoid
- ReLU (Rectified Linear Unit)
- Tanh (Hyperbolic Tangent)
- Linear
- Softmax (for future multi-class classification)

All activation functions include their derivatives for backpropagation.

### Neural Network Components
- **Perceptron**: Single-layer perceptron with configurable activation functions
- **Loss Functions**: Mean Squared Error (MSE)
- **Training**: Gradient descent with configurable:
  - Learning rate
  - Batch size
  - Maximum epochs
  - Convergence tolerance
  - Verbose output options

### Dataset Handling
- Custom dataset structure with feature matrix and target vector
- Pre-built logic gate datasets (AND, OR, XOR)
- Easy extension for custom datasets

## Project Structure

```
nn_c/
├── src/
│   ├── activations/       # Activation functions and derivatives
│   ├── data/             # Dataset creation and management
│   ├── linalg/           # Linear algebra (matrix and vector operations)
│   ├── nn/               # Neural network components (perceptron, loss)
│   ├── training/         # Training algorithms (gradient descent)
│   └── main.c            # Example application
├── tests/                # Unit tests for all components
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

The default application demonstrates perceptron learning on logic gates:
- **AND Gate**: Successfully learns (linearly separable)
- **OR Gate**: Successfully learns (linearly separable)
- **XOR Gate**: Fails to converge (not linearly separable)

The XOR failure demonstrates the fundamental limitation of single-layer perceptrons, motivating the need for multi-layer architectures.

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

### Phase 3: Multi-Layer Perceptron (Planned)
- Hidden layers
- Backpropagation algorithm
- More complex datasets
- Network architecture configuration

## Code Style

The project follows a consistent C coding style:
- Snake case for functions and variables
- Descriptive naming conventions
- Clear separation of concerns
- Comprehensive comments
- Formatted with clang-format

## Memory Management

All dynamically allocated structures include corresponding `_free()` functions. The project is designed to be leak-free (verify with `make memcheck`).

## License

This is an educational project. Feel free to use and modify as needed.

## Contributing

This is a personal learning project, but suggestions and improvements are welcome.

## Acknowledgments

Built from scratch to understand neural networks at a fundamental level, without relying on high-level machine learning frameworks.
