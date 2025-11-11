/*
 * test_runner.h - Central header for all test suite runners
 */

#ifndef TEST_RUNNER_H
#define TEST_RUNNER_H
#include "../src/linalg/matrix.h"
#include "../src/linalg/vector.h"
#include <math.h>

// Shared constants.
#define EPSILON 1e-6
#define TEST_PASSED printf("  ✓ %s passed\n", __func__)
#define TEST_FAILED printf("  ✗ %s FAILED\n", __func__)

// Shared helper functions
static int float_equals(float a, float b) {
    return fabs(a - b) < EPSILON;
}

static int matrix_equals(const Matrix *a, const Matrix *b) {
    if (a->rows != b->rows || a->cols != b->cols) {
        return 0;
    }
    for (int i = 0; i < a->rows * a->cols; i++) {
        if (!float_equals(a->data[i], b->data[i])) {
            return 0;
        }
    }
    return 1;
}

static int vector_equals(const Vector *a, const Vector *b) {
    if (a->size != b->size) {
        return 0;
    }
    for (int i = 0; i < a->size; i++) {
        if (!float_equals(a->data[i], b->data[i])) {
            return 0;
        }
    }
    return 1;
}

// Vector tests
void run_vector_tests(void);

// Matrix tests
void run_matrix_tests(void);

// Activation tests
void run_activations_tests(void);

#endif /* TEST_RUNNER_H */
