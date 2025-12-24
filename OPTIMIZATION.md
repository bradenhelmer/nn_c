# Optimizations

This is a timeline of the various optimizations made to speed up the training of my convolutional neural network on the MNIST dataset.

## Neural Net Architecture

The network trained on the MNIST composes of 7 layers:

1. 2D Convolutional Layer (Input Tensor (1x28x28))
  - Input channels: 1
  - Output channels: 32
  - Kernel size: 5x5
  - Stride: 1
  - Padding: 2

2. ReLU Layer

3. Max Pooling Layer:
  - Pool size: 2
  - Stride: 2

4. Flatten Layer

5. Linear Layer:
  - Input: 6272
  - Output: 128

6. ReLU Layer

7. Linear Layer:
  - Input: 128
  - Output: 10

Loss Function: Softmax Cross Entropy
