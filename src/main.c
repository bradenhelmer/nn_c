/*
 * main.c - Neural network application entry point
 */

#include <stdlib.h>
#include <time.h>

// External example functions

// Perceptron
extern void perceptron_learning_logic_gates(void);

// Basic MLP
extern void mlp_learning_xor(void);
extern void mlp_learning_xor_batched(void);

// MNIST
extern void mnist_sgd();
extern void mnist_momentum();
extern void mnist_adam();

int main() {
    srand(time(NULL));
    // perceptron_learning_logic_gates();
    // mlp_learning_xor();
    // mlp_learning_xor_batched();
    mnist_sgd();
    mnist_momentum();
    mnist_adam();
    return 0;
}
