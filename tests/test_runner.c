/*
 * test_runner.c - Test suite runner for neural network components
 */

#include "test_runner.h"
#include <stdio.h>

int main() {
    printf("\n");
    printf("=====================================\n");
    printf("   Neural Network Test Suite\n");
    printf("=====================================\n");

    // Run activation tests
    run_activations_tests();

    // Run perceptron tests
    run_perceptron_tests();

    // Run loss tests
    run_loss_tests();

    // Run batch tests
    run_batch_tests();

    // Run optimizer tests
    run_optimizer_tests();

    // Run tensor tests
    run_tensor_tests();

    // Run conv layer tests
    run_conv_tests();

    // Run pool layer tests
    run_pool_tests();

    printf("\n");
    printf("=====================================\n");
    printf("   All Tests Passed Successfully!\n");
    printf("=====================================\n");
    printf("\n");

    return 0;
}
