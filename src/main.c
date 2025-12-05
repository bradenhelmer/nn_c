/*
 * main.c - Neural network application entry point
 */

#include <stdlib.h>
#include <time.h>

// External example functions
extern void perceptron_learning_logic_gates(void);
extern void mlp_learning_xor(void);
extern void mlp_learning_xor_batched(void);

int main() {
    srand(time(NULL));
    perceptron_learning_logic_gates();
    mlp_learning_xor();
    mlp_learning_xor_batched();
    return 0;
}
