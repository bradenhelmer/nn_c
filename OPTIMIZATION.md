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

## Base Run

### Training Configuration

The training code can be found in the `mnist_conv` function in [mnist_examples.c](./src/examples/mnist_examples.c)

- Method: Batched gradient-descent
- Optimzer: ADAM (learning_rate: 0.001, beta1: 0.9, beta2: 0.999, epsilon: 1e-8)
- Scheduler: Cosine Annealing (initial_learning_rate: 0.001, min_learning_rate: 1e-5, max_timestep: 20)
- Max Epochs: 10
- L2 Lambda regularization: 0.0001
- Learning Rate: 0.5

### Compiler Information

For a baseline profile, we can compile with flags for a release build:
```sh
clang -std=c99 -Wall -Wextra -O3 -march=native -flto -DNDEBUG ...
```

### Timer

We'll use a simple POSIX timer to time the baseline training loop:

```c
#define _POSIX_C_SOURCE 199309L
#include <time.h>

typedef struct {
    struct timespec start;
    double elapsed;
} Timer;

static inline void timer_start(Timer *t) {
    clock_gettime(CLOCK_MONOTONIC, &t->start);
}

static inline void timer_stop(Timer *t) {
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    t->elapsed = (end.tv_sec - t->start.tv_sec) + (end.tv_nsec - t->start.tv_nsec) * 1e-9;
}
```

Integrated with the training logic:

```c
Timer training_timer = {0};
timer_start(&training_timer);
TrainingResult *mnist_conv_result = train_nn_batch_opt(mnist_conv, mnist_train, NULL, &config);
timer_stop(&training_timer);
printf("Training took: %f seconds\n", training_timer.elapsed);
```

### Baseline Results

```
Epoch 0: loss=0.2594, accuracy=92.58%
Epoch 1: loss=0.0983, accuracy=97.04%
Epoch 2: loss=0.0757, accuracy=97.64%
Epoch 3: loss=0.0674, accuracy=97.91%
Epoch 4: loss=0.0585, accuracy=98.21%
Epoch 5: loss=0.0477, accuracy=98.50%
Epoch 6: loss=0.0418, accuracy=98.71%
Epoch 7: loss=0.0412, accuracy=98.66%
Epoch 8: loss=0.0408, accuracy=98.71%
Epoch 9: loss=0.0331, accuracy=99.00%
Training took: 1556.741196 seconds

MNIST Batched Convolutional NN training with ADAM/cosine annealing stopped at 10 epochs
Final accuracy: 99.00%

Testing MNIST test images on batched convolutional NN with cosine annealing and L2 regularization:

Final images correctly classified: 9868
```

Some pretty promising results from a model perspective, but that was slow.
Just under 26 minutes total time, around 2 and a half minutes per epoch.
