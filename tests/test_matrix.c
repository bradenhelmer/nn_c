/*
 * test_matrix.c - Comprehensive tests for matrix operations
 */

#include "../src/linalg/matrix.h"
#include "../src/linalg/vector.h"
#include "test_runner.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// =============================================================================
// Creation and Destruction Tests
// =============================================================================

void test_matrix_create() {
    Matrix *m = matrix_create(3, 4);
    assert(m != NULL);
    assert(m->rows == 3);
    assert(m->cols == 4);
    assert(m->data != NULL);
    matrix_free(m);
    TEST_PASSED;
}

void test_matrix_zeros() {
    Matrix *m = matrix_zeros(2, 3);
    assert(m != NULL);
    assert(m->rows == 2);
    assert(m->cols == 3);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            assert(float_equals(matrix_get(m, i, j), 0.0f));
        }
    }
    matrix_free(m);
    TEST_PASSED;
}

void test_matrix_ones() {
    Matrix *m = matrix_ones(2, 2);
    assert(m != NULL);
    assert(m->rows == 2);
    assert(m->cols == 2);
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            assert(float_equals(matrix_get(m, i, j), 1.0f));
        }
    }
    matrix_free(m);
    TEST_PASSED;
}

void test_matrix_random() {
    Matrix *m = matrix_random(3, 3, -10.0f, 10.0f);
    assert(m != NULL);
    assert(m->rows == 3);
    assert(m->cols == 3);

    // Check all values are within bounds
    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            float val = matrix_get(m, i, j);
            assert(val >= -10.0f);
            assert(val <= 10.0f);
        }
    }

    matrix_free(m);
    TEST_PASSED;
}

void test_matrix_identity() {
    Matrix *m = matrix_identity(3);
    assert(m != NULL);
    assert(m->rows == 3);
    assert(m->cols == 3);

    for (int i = 0; i < m->rows; i++) {
        for (int j = 0; j < m->cols; j++) {
            if (i == j) {
                assert(float_equals(matrix_get(m, i, j), 1.0f));
            } else {
                assert(float_equals(matrix_get(m, i, j), 0.0f));
            }
        }
    }

    matrix_free(m);
    TEST_PASSED;
}

// =============================================================================
// Core Linear Algebra Tests
// =============================================================================

void test_matrix_multiply() {
    // Create 2x3 matrix
    Matrix *a = matrix_create(2, 3);
    matrix_set(a, 0, 0, 1.0f);
    matrix_set(a, 0, 1, 2.0f);
    matrix_set(a, 0, 2, 3.0f);
    matrix_set(a, 1, 0, 4.0f);
    matrix_set(a, 1, 1, 5.0f);
    matrix_set(a, 1, 2, 6.0f);

    // Create 3x2 matrix
    Matrix *b = matrix_create(3, 2);
    matrix_set(b, 0, 0, 7.0f);
    matrix_set(b, 0, 1, 8.0f);
    matrix_set(b, 1, 0, 9.0f);
    matrix_set(b, 1, 1, 10.0f);
    matrix_set(b, 2, 0, 11.0f);
    matrix_set(b, 2, 1, 12.0f);

    // Result should be 2x2
    Matrix *result = matrix_create(2, 2);
    matrix_multiply(result, a, b);

    // Expected: [[58, 64], [139, 154]]
    assert(float_equals(matrix_get(result, 0, 0), 58.0f));
    assert(float_equals(matrix_get(result, 0, 1), 64.0f));
    assert(float_equals(matrix_get(result, 1, 0), 139.0f));
    assert(float_equals(matrix_get(result, 1, 1), 154.0f));

    matrix_free(a);
    matrix_free(b);
    matrix_free(result);
    TEST_PASSED;
}

void test_matrix_multiply_identity() {
    Matrix *a = matrix_create(2, 2);
    matrix_set(a, 0, 0, 1.0f);
    matrix_set(a, 0, 1, 2.0f);
    matrix_set(a, 1, 0, 3.0f);
    matrix_set(a, 1, 1, 4.0f);

    Matrix *identity = matrix_identity(2);
    Matrix *result = matrix_create(2, 2);

    matrix_multiply(result, a, identity);

    // A * I should equal A
    assert(matrix_equals(a, result));

    matrix_free(a);
    matrix_free(identity);
    matrix_free(result);
    TEST_PASSED;
}

void test_matrix_transpose() {
    Matrix *m = matrix_create(2, 3);
    matrix_set(m, 0, 0, 1.0f);
    matrix_set(m, 0, 1, 2.0f);
    matrix_set(m, 0, 2, 3.0f);
    matrix_set(m, 1, 0, 4.0f);
    matrix_set(m, 1, 1, 5.0f);
    matrix_set(m, 1, 2, 6.0f);

    Matrix *result = matrix_create(2, 3); // Will be resized to 3x2
    matrix_transpose(result, m);

    assert(result->rows == 3);
    assert(result->cols == 2);
    assert(float_equals(matrix_get(result, 0, 0), 1.0f));
    assert(float_equals(matrix_get(result, 1, 0), 2.0f));
    assert(float_equals(matrix_get(result, 2, 0), 3.0f));
    assert(float_equals(matrix_get(result, 0, 1), 4.0f));
    assert(float_equals(matrix_get(result, 1, 1), 5.0f));
    assert(float_equals(matrix_get(result, 2, 1), 6.0f));

    matrix_free(m);
    matrix_free(result);
    TEST_PASSED;
}

void test_matrix_add() {
    Matrix *a = matrix_create(2, 2);
    Matrix *b = matrix_create(2, 2);
    Matrix *result = matrix_create(2, 2);

    matrix_set(a, 0, 0, 1.0f);
    matrix_set(a, 0, 1, 2.0f);
    matrix_set(a, 1, 0, 3.0f);
    matrix_set(a, 1, 1, 4.0f);

    matrix_set(b, 0, 0, 5.0f);
    matrix_set(b, 0, 1, 6.0f);
    matrix_set(b, 1, 0, 7.0f);
    matrix_set(b, 1, 1, 8.0f);

    matrix_add(result, a, b);

    assert(float_equals(matrix_get(result, 0, 0), 6.0f));
    assert(float_equals(matrix_get(result, 0, 1), 8.0f));
    assert(float_equals(matrix_get(result, 1, 0), 10.0f));
    assert(float_equals(matrix_get(result, 1, 1), 12.0f));

    matrix_free(a);
    matrix_free(b);
    matrix_free(result);
    TEST_PASSED;
}

void test_matrix_subtract() {
    Matrix *a = matrix_create(2, 2);
    Matrix *b = matrix_create(2, 2);
    Matrix *result = matrix_create(2, 2);

    matrix_set(a, 0, 0, 10.0f);
    matrix_set(a, 0, 1, 8.0f);
    matrix_set(a, 1, 0, 6.0f);
    matrix_set(a, 1, 1, 4.0f);

    matrix_set(b, 0, 0, 1.0f);
    matrix_set(b, 0, 1, 2.0f);
    matrix_set(b, 1, 0, 3.0f);
    matrix_set(b, 1, 1, 4.0f);

    matrix_subtract(result, a, b);

    assert(float_equals(matrix_get(result, 0, 0), 9.0f));
    assert(float_equals(matrix_get(result, 0, 1), 6.0f));
    assert(float_equals(matrix_get(result, 1, 0), 3.0f));
    assert(float_equals(matrix_get(result, 1, 1), 0.0f));

    matrix_free(a);
    matrix_free(b);
    matrix_free(result);
    TEST_PASSED;
}

void test_matrix_scale() {
    Matrix *m = matrix_create(2, 2);
    Matrix *result = matrix_create(2, 2);

    matrix_set(m, 0, 0, 1.0f);
    matrix_set(m, 0, 1, 2.0f);
    matrix_set(m, 1, 0, 3.0f);
    matrix_set(m, 1, 1, 4.0f);

    matrix_scale(result, m, 2.5f);

    assert(float_equals(matrix_get(result, 0, 0), 2.5f));
    assert(float_equals(matrix_get(result, 0, 1), 5.0f));
    assert(float_equals(matrix_get(result, 1, 0), 7.5f));
    assert(float_equals(matrix_get(result, 1, 1), 10.0f));

    matrix_free(m);
    matrix_free(result);
    TEST_PASSED;
}

// =============================================================================
// Element-wise Operations Tests
// =============================================================================

void test_matrix_multiply_elementwise() {
    Matrix *a = matrix_create(2, 2);
    Matrix *b = matrix_create(2, 2);
    Matrix *result = matrix_create(2, 2);

    matrix_set(a, 0, 0, 2.0f);
    matrix_set(a, 0, 1, 3.0f);
    matrix_set(a, 1, 0, 4.0f);
    matrix_set(a, 1, 1, 5.0f);

    matrix_set(b, 0, 0, 6.0f);
    matrix_set(b, 0, 1, 7.0f);
    matrix_set(b, 1, 0, 8.0f);
    matrix_set(b, 1, 1, 9.0f);

    matrix_multiply_elementwise(result, a, b);

    assert(float_equals(matrix_get(result, 0, 0), 12.0f));
    assert(float_equals(matrix_get(result, 0, 1), 21.0f));
    assert(float_equals(matrix_get(result, 1, 0), 32.0f));
    assert(float_equals(matrix_get(result, 1, 1), 45.0f));

    matrix_free(a);
    matrix_free(b);
    matrix_free(result);
    TEST_PASSED;
}

void test_matrix_add_scalar() {
    Matrix *m = matrix_create(2, 2);
    Matrix *result = matrix_create(2, 2);

    matrix_set(m, 0, 0, 1.0f);
    matrix_set(m, 0, 1, 2.0f);
    matrix_set(m, 1, 0, 3.0f);
    matrix_set(m, 1, 1, 4.0f);

    matrix_add_scalar(result, m, 5.0f);

    assert(float_equals(matrix_get(result, 0, 0), 6.0f));
    assert(float_equals(matrix_get(result, 0, 1), 7.0f));
    assert(float_equals(matrix_get(result, 1, 0), 8.0f));
    assert(float_equals(matrix_get(result, 1, 1), 9.0f));

    matrix_free(m);
    matrix_free(result);
    TEST_PASSED;
}

// =============================================================================
// Matrix-Vector Operations Tests
// =============================================================================

void test_matrix_vector_multiply() {
    Matrix *m = matrix_create(2, 3);
    matrix_set(m, 0, 0, 1.0f);
    matrix_set(m, 0, 1, 2.0f);
    matrix_set(m, 0, 2, 3.0f);
    matrix_set(m, 1, 0, 4.0f);
    matrix_set(m, 1, 1, 5.0f);
    matrix_set(m, 1, 2, 6.0f);

    Vector *v = vector_create(3);
    v->data[0] = 2.0f;
    v->data[1] = 3.0f;
    v->data[2] = 1.0f;

    Vector *result = vector_create(2);
    matrix_vector_multiply(result, m, v);

    // Expected: [1*2+2*3+3*1, 4*2+5*3+6*1] = [11, 29]
    assert(float_equals(result->data[0], 11.0f));
    assert(float_equals(result->data[1], 29.0f));

    matrix_free(m);
    vector_free(v);
    vector_free(result);
    TEST_PASSED;
}

void test_matrix_add_vector() {
    Matrix *m = matrix_create(2, 3);
    matrix_set(m, 0, 0, 1.0f);
    matrix_set(m, 0, 1, 2.0f);
    matrix_set(m, 0, 2, 3.0f);
    matrix_set(m, 1, 0, 4.0f);
    matrix_set(m, 1, 1, 5.0f);
    matrix_set(m, 1, 2, 6.0f);

    Vector *v = vector_create(3);
    v->data[0] = 10.0f;
    v->data[1] = 20.0f;
    v->data[2] = 30.0f;

    Matrix *result = matrix_create(2, 3);
    matrix_add_vector(result, m, v);

    // Each row should have the vector added
    assert(float_equals(matrix_get(result, 0, 0), 11.0f));
    assert(float_equals(matrix_get(result, 0, 1), 22.0f));
    assert(float_equals(matrix_get(result, 0, 2), 33.0f));
    assert(float_equals(matrix_get(result, 1, 0), 14.0f));
    assert(float_equals(matrix_get(result, 1, 1), 25.0f));
    assert(float_equals(matrix_get(result, 1, 2), 36.0f));

    matrix_free(m);
    vector_free(v);
    matrix_free(result);
    TEST_PASSED;
}

// =============================================================================
// Utility Functions Tests
// =============================================================================

void test_matrix_copy() {
    Matrix *src = matrix_create(2, 3);
    matrix_set(src, 0, 0, 1.5f);
    matrix_set(src, 0, 1, 2.5f);
    matrix_set(src, 0, 2, 3.5f);
    matrix_set(src, 1, 0, 4.5f);
    matrix_set(src, 1, 1, 5.5f);
    matrix_set(src, 1, 2, 6.5f);

    Matrix *dest = matrix_create(2, 3);
    matrix_copy(dest, src);

    assert(matrix_equals(src, dest));

    matrix_free(src);
    matrix_free(dest);
    TEST_PASSED;
}

void test_matrix_get_set() {
    Matrix *m = matrix_create(3, 3);

    matrix_set(m, 0, 0, 1.0f);
    matrix_set(m, 1, 1, 2.0f);
    matrix_set(m, 2, 2, 3.0f);

    assert(float_equals(matrix_get(m, 0, 0), 1.0f));
    assert(float_equals(matrix_get(m, 1, 1), 2.0f));
    assert(float_equals(matrix_get(m, 2, 2), 3.0f));

    matrix_free(m);
    TEST_PASSED;
}

void test_matrix_print() {
    Matrix *m = matrix_create(2, 3);
    matrix_set(m, 0, 0, 1.0f);
    matrix_set(m, 0, 1, 2.0f);
    matrix_set(m, 0, 2, 3.0f);
    matrix_set(m, 1, 0, 4.0f);
    matrix_set(m, 1, 1, 5.0f);
    matrix_set(m, 1, 2, 6.0f);

    printf("  Testing matrix_print (visual check):\n  ");
    matrix_print(m);
    printf("\n");

    matrix_free(m);
    TEST_PASSED;
}

// =============================================================================
// Special Operations for Neural Networks Tests
// =============================================================================

void test_matrix_sum_rows() {
    Matrix *m = matrix_create(3, 4);
    // Row 0: [1, 2, 3, 4] -> sum = 10
    // Row 1: [5, 6, 7, 8] -> sum = 26
    // Row 2: [9, 10, 11, 12] -> sum = 42
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            matrix_set(m, i, j, (float)(i * 4 + j + 1));
        }
    }

    Vector *result = vector_create(3);
    matrix_sum_rows(result, m);

    assert(float_equals(result->data[0], 10.0f));
    assert(float_equals(result->data[1], 26.0f));
    assert(float_equals(result->data[2], 42.0f));

    matrix_free(m);
    vector_free(result);
    TEST_PASSED;
}

void test_matrix_sum_cols() {
    Matrix *m = matrix_create(3, 4);
    // Col 0: [1, 5, 9] -> sum = 15
    // Col 1: [2, 6, 10] -> sum = 18
    // Col 2: [3, 7, 11] -> sum = 21
    // Col 3: [4, 8, 12] -> sum = 24
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 4; j++) {
            matrix_set(m, i, j, (float)(i * 4 + j + 1));
        }
    }

    Vector *result = vector_create(4);
    matrix_sum_cols(result, m);

    assert(float_equals(result->data[0], 15.0f));
    assert(float_equals(result->data[1], 18.0f));
    assert(float_equals(result->data[2], 21.0f));
    assert(float_equals(result->data[3], 24.0f));

    matrix_free(m);
    vector_free(result);
    TEST_PASSED;
}

// =============================================================================
// Edge Cases Tests
// =============================================================================

void test_matrix_1x1() {
    Matrix *m = matrix_ones(1, 1);
    assert(float_equals(matrix_get(m, 0, 0), 1.0f));

    Matrix *result = matrix_create(1, 1);
    matrix_scale(result, m, 5.0f);
    assert(float_equals(matrix_get(result, 0, 0), 5.0f));

    matrix_free(m);
    matrix_free(result);
    TEST_PASSED;
}

void test_matrix_add_in_place() {
    Matrix *a = matrix_create(2, 2);
    Matrix *b = matrix_create(2, 2);

    matrix_set(a, 0, 0, 1.0f);
    matrix_set(a, 0, 1, 2.0f);
    matrix_set(a, 1, 0, 3.0f);
    matrix_set(a, 1, 1, 4.0f);

    matrix_set(b, 0, 0, 5.0f);
    matrix_set(b, 0, 1, 6.0f);
    matrix_set(b, 1, 0, 7.0f);
    matrix_set(b, 1, 1, 8.0f);

    // Use 'a' as both input and output
    matrix_add(a, a, b);

    assert(float_equals(matrix_get(a, 0, 0), 6.0f));
    assert(float_equals(matrix_get(a, 0, 1), 8.0f));
    assert(float_equals(matrix_get(a, 1, 0), 10.0f));
    assert(float_equals(matrix_get(a, 1, 1), 12.0f));

    matrix_free(a);
    matrix_free(b);
    TEST_PASSED;
}

void test_matrix_transpose_square() {
    Matrix *m = matrix_create(2, 2);
    matrix_set(m, 0, 0, 1.0f);
    matrix_set(m, 0, 1, 2.0f);
    matrix_set(m, 1, 0, 3.0f);
    matrix_set(m, 1, 1, 4.0f);

    Matrix *result = matrix_create(2, 2);
    matrix_transpose(result, m);

    assert(float_equals(matrix_get(result, 0, 0), 1.0f));
    assert(float_equals(matrix_get(result, 0, 1), 3.0f));
    assert(float_equals(matrix_get(result, 1, 0), 2.0f));
    assert(float_equals(matrix_get(result, 1, 1), 4.0f));

    matrix_free(m);
    matrix_free(result);
    TEST_PASSED;
}

// =============================================================================
// Test Runner
// =============================================================================

void run_matrix_tests(void) {
    printf("\n=== Matrix Creation and Destruction Tests ===\n");
    test_matrix_create();
    test_matrix_zeros();
    test_matrix_ones();
    test_matrix_random();
    test_matrix_identity();

    printf("\n=== Matrix Core Linear Algebra Tests ===\n");
    test_matrix_multiply();
    test_matrix_multiply_identity();
    test_matrix_transpose();
    test_matrix_add();
    test_matrix_subtract();
    test_matrix_scale();

    printf("\n=== Matrix Element-wise Operations Tests ===\n");
    test_matrix_multiply_elementwise();
    test_matrix_add_scalar();

    printf("\n=== Matrix-Vector Operations Tests ===\n");
    test_matrix_vector_multiply();
    test_matrix_add_vector();

    printf("\n=== Matrix Utility Functions Tests ===\n");
    test_matrix_copy();
    test_matrix_get_set();
    test_matrix_print();

    printf("\n=== Matrix Special Operations Tests ===\n");
    test_matrix_sum_rows();
    test_matrix_sum_cols();

    printf("\n=== Matrix Edge Cases Tests ===\n");
    test_matrix_1x1();
    test_matrix_add_in_place();
    test_matrix_transpose_square();

    printf("\n=== All Matrix Tests Passed! ===\n");
}
