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

    // Run vector tests
    run_vector_tests();

    // Run matrix tests
    run_matrix_tests();

    // Run activation tests
    run_activations_tests();

    // Run perceptron tests
    run_perceptron_tests();

    // Run loss tests
    run_loss_tests();

    printf("\n");
    printf("=====================================\n");
    printf("   All Tests Passed Successfully!\n");
    printf("=====================================\n");
    printf("\n");

    return 0;
}
