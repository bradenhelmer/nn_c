/*
 * main.c - Neural network application entry point
 */

#include "tensor/gpu_tensor.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// External example functions

// Perceptron
extern void perceptron_learning_logic_gates(void);

// Basic linear layer NN
extern void nn_learning_xor(void);
extern void nn_learning_xor_batched(void);

// MNIST
extern void mnist_sgd();
extern void mnist_momentum();
extern void mnist_adam();
extern void mnist_aggressive();
extern void mnist_conv();

static void gpu_test() {
    GPUTensor *gpu_t = gpu_tensor_create(2, (int[]){128, 128, 0, 0});
    printf("Capacity: %zu", gpu_t->capacity);
    gpu_tensor_free(gpu_t);
}

int main() {
    // srand(time(NULL));
    // perceptron_learning_logic_gates();
    // nn_learning_xor();
    // nn_learning_xor_batched();
    // mnist_sgd();
    // mnist_momentum();
    // mnist_adam();
    // mnist_aggressive();
    // mnist_conv();
    gpu_test();
    return 0;
}
