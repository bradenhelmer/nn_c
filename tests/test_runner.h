/*
 * test_runner.h - Central header for all test suite runners
 */

#ifndef TEST_RUNNER_H
#define TEST_RUNNER_H
#include "../src/linalg/matrix.h"
#include "../src/linalg/vector.h"
#include "../src/utils/utils.h"

// Shared constants.
#define TEST_PASSED printf("  ✓ %s passed\n", __func__)
#define TEST_FAILED printf("  ✗ %s FAILED\n", __func__)

// Vector tests
void run_vector_tests(void);

// Matrix tests
void run_matrix_tests(void);

// Activation tests
void run_activations_tests(void);

// Perceptron tests
void run_perceptron_tests(void);

// Loss tests
void run_loss_tests(void);

// Batch tests
void run_batch_tests(void);

// Optimizer tests
void run_optimizer_tests(void);

#endif /* TEST_RUNNER_H */
