/*
 * test_vector.c - Comprehensive tests for vector operations
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

void test_vector_create() {
    Vector *v = vector_create(5);
    assert(v != NULL);
    assert(v->size == 5);
    assert(v->data != NULL);
    vector_free(v);
    TEST_PASSED;
}

void test_vector_zeros() {
    Vector *v = vector_zeros(4);
    assert(v != NULL);
    assert(v->size == 4);
    for (int i = 0; i < v->size; i++) {
        assert(float_equals(v->data[i], 0.0f));
    }
    vector_free(v);
    TEST_PASSED;
}

void test_vector_ones() {
    Vector *v = vector_ones(3);
    assert(v != NULL);
    assert(v->size == 3);
    for (int i = 0; i < v->size; i++) {
        assert(float_equals(v->data[i], 1.0f));
    }
    vector_free(v);
    TEST_PASSED;
}

void test_vector_random() {
    Vector *v = vector_random(10, -5.0f, 5.0f);
    assert(v != NULL);
    assert(v->size == 10);

    // Check all values are within bounds
    for (int i = 0; i < v->size; i++) {
        assert(v->data[i] >= -5.0f);
        assert(v->data[i] <= 5.0f);
    }

    // Check that not all values are the same (probabilistically should pass)
    int all_same = 1;
    for (int i = 1; i < v->size; i++) {
        if (!float_equals(v->data[i], v->data[0])) {
            all_same = 0;
            break;
        }
    }
    assert(all_same == 0);

    vector_free(v);
    TEST_PASSED;
}

// =============================================================================
// Basic Operations Tests
// =============================================================================

void test_vector_add() {
    Vector *a = vector_create(3);
    Vector *b = vector_create(3);
    Vector *result = vector_create(3);

    a->data[0] = 1.0f;
    a->data[1] = 2.0f;
    a->data[2] = 3.0f;
    b->data[0] = 4.0f;
    b->data[1] = 5.0f;
    b->data[2] = 6.0f;

    vector_add(result, a, b);

    assert(float_equals(result->data[0], 5.0f));
    assert(float_equals(result->data[1], 7.0f));
    assert(float_equals(result->data[2], 9.0f));

    vector_free(a);
    vector_free(b);
    vector_free(result);
    TEST_PASSED;
}

void test_vector_subtract() {
    Vector *a = vector_create(3);
    Vector *b = vector_create(3);
    Vector *result = vector_create(3);

    a->data[0] = 10.0f;
    a->data[1] = 8.0f;
    a->data[2] = 6.0f;
    b->data[0] = 4.0f;
    b->data[1] = 3.0f;
    b->data[2] = 2.0f;

    vector_subtract(result, a, b);

    assert(float_equals(result->data[0], 6.0f));
    assert(float_equals(result->data[1], 5.0f));
    assert(float_equals(result->data[2], 4.0f));

    vector_free(a);
    vector_free(b);
    vector_free(result);
    TEST_PASSED;
}

void test_vector_scale() {
    Vector *v = vector_create(3);
    Vector *result = vector_create(3);

    v->data[0] = 1.0f;
    v->data[1] = 2.0f;
    v->data[2] = 3.0f;

    vector_scale(result, v, 2.5f);

    assert(float_equals(result->data[0], 2.5f));
    assert(float_equals(result->data[1], 5.0f));
    assert(float_equals(result->data[2], 7.5f));

    vector_free(v);
    vector_free(result);
    TEST_PASSED;
}

void test_vector_dot() {
    Vector *a = vector_create(3);
    Vector *b = vector_create(3);

    a->data[0] = 1.0f;
    a->data[1] = 2.0f;
    a->data[2] = 3.0f;
    b->data[0] = 4.0f;
    b->data[1] = 5.0f;
    b->data[2] = 6.0f;

    float result = vector_dot(a, b);
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    assert(float_equals(result, 32.0f));

    vector_free(a);
    vector_free(b);
    TEST_PASSED;
}

void test_vector_dot_orthogonal() {
    Vector *a = vector_create(2);
    Vector *b = vector_create(2);

    a->data[0] = 1.0f;
    a->data[1] = 0.0f;
    b->data[0] = 0.0f;
    b->data[1] = 1.0f;

    float result = vector_dot(a, b);
    assert(float_equals(result, 0.0f)); // Orthogonal vectors

    vector_free(a);
    vector_free(b);
    TEST_PASSED;
}

// =============================================================================
// Element-wise Operations Tests
// =============================================================================

void test_vector_multiply() {
    Vector *a = vector_create(3);
    Vector *b = vector_create(3);
    Vector *result = vector_create(3);

    a->data[0] = 2.0f;
    a->data[1] = 3.0f;
    a->data[2] = 4.0f;
    b->data[0] = 5.0f;
    b->data[1] = 6.0f;
    b->data[2] = 7.0f;

    vector_elementwise_multiply(result, a, b);

    assert(float_equals(result->data[0], 10.0f));
    assert(float_equals(result->data[1], 18.0f));
    assert(float_equals(result->data[2], 28.0f));

    vector_free(a);
    vector_free(b);
    vector_free(result);
    TEST_PASSED;
}

void test_vector_divide() {
    Vector *a = vector_create(3);
    Vector *b = vector_create(3);
    Vector *result = vector_create(3);

    a->data[0] = 10.0f;
    a->data[1] = 20.0f;
    a->data[2] = 30.0f;
    b->data[0] = 2.0f;
    b->data[1] = 4.0f;
    b->data[2] = 5.0f;

    vector_elementwise_divide(result, a, b);

    assert(float_equals(result->data[0], 5.0f));
    assert(float_equals(result->data[1], 5.0f));
    assert(float_equals(result->data[2], 6.0f));

    vector_free(a);
    vector_free(b);
    vector_free(result);
    TEST_PASSED;
}

// =============================================================================
// Other Functions Tests
// =============================================================================

void test_vector_outer_product() {
    Vector *a = vector_create(3);
    Vector *b = vector_create(2);
    Matrix *result = matrix_create(3, 2);

    a->data[0] = 10.0f;
    a->data[1] = 20.0f;
    a->data[2] = 30.0f;
    b->data[0] = 2.0f;
    b->data[1] = 4.0f;

    vector_outer_product(result, a, b);

    assert(float_equals(matrix_get(result, 0, 0), 20.0f));
    assert(float_equals(matrix_get(result, 0, 1), 40.0f));
    assert(float_equals(matrix_get(result, 1, 0), 40.0f));
    assert(float_equals(matrix_get(result, 1, 1), 80.0f));
    assert(float_equals(matrix_get(result, 2, 0), 60.0f));
    assert(float_equals(matrix_get(result, 2, 1), 120.0f));

    vector_free(a);
    vector_free(b);
    matrix_free(result);
    TEST_PASSED;
}

// =============================================================================
// Utility Functions Tests
// =============================================================================

void test_vector_copy() {
    Vector *src = vector_create(4);
    Vector *dest = vector_create(4);

    src->data[0] = 1.5f;
    src->data[1] = 2.5f;
    src->data[2] = 3.5f;
    src->data[3] = 4.5f;

    vector_copy(dest, src);

    assert(vector_equals(src, dest));

    vector_free(src);
    vector_free(dest);
    TEST_PASSED;
}

void test_vector_sum() {
    Vector *v = vector_create(4);

    v->data[0] = 1.0f;
    v->data[1] = 2.0f;
    v->data[2] = 3.0f;
    v->data[3] = 4.0f;

    float sum = vector_sum(v);
    assert(float_equals(sum, 10.0f));

    vector_free(v);
    TEST_PASSED;
}

void test_vector_sum_negative() {
    Vector *v = vector_create(3);

    v->data[0] = -5.0f;
    v->data[1] = 10.0f;
    v->data[2] = -3.0f;

    float sum = vector_sum(v);
    assert(float_equals(sum, 2.0f));

    vector_free(v);
    TEST_PASSED;
}

void test_vector_min() {
    Vector *v = vector_create(5);

    v->data[0] = 3.0f;
    v->data[1] = -2.0f;
    v->data[2] = 7.0f;
    v->data[3] = 1.0f;
    v->data[4] = -5.0f;

    float min = vector_min(v);
    assert(float_equals(min, -5.0f));

    vector_free(v);
    TEST_PASSED;
}

void test_vector_max() {
    Vector *v = vector_create(5);

    v->data[0] = 3.0f;
    v->data[1] = -2.0f;
    v->data[2] = 7.0f;
    v->data[3] = 1.0f;
    v->data[4] = -5.0f;

    float max = vector_max(v);
    assert(float_equals(max, 7.0f));

    vector_free(v);
    TEST_PASSED;
}

void test_vector_print() {
    Vector *v = vector_create(3);
    v->data[0] = 1.0f;
    v->data[1] = 2.0f;
    v->data[2] = 3.0f;

    printf("  Testing vector_print (visual check): ");
    vector_print(v);
    printf("\n");

    vector_free(v);
    TEST_PASSED;
}

// =============================================================================
// Edge Cases Tests
// =============================================================================

void test_vector_single_element() {
    Vector *v = vector_ones(1);
    assert(v->size == 1);
    assert(float_equals(v->data[0], 1.0f));
    assert(float_equals(vector_sum(v), 1.0f));
    assert(float_equals(vector_min(v), 1.0f));
    assert(float_equals(vector_max(v), 1.0f));

    vector_free(v);
    TEST_PASSED;
}

void test_vector_add_in_place() {
    Vector *a = vector_create(3);
    Vector *b = vector_create(3);

    a->data[0] = 1.0f;
    a->data[1] = 2.0f;
    a->data[2] = 3.0f;
    b->data[0] = 4.0f;
    b->data[1] = 5.0f;
    b->data[2] = 6.0f;

    // Use 'a' as both input and output
    vector_add(a, a, b);

    assert(float_equals(a->data[0], 5.0f));
    assert(float_equals(a->data[1], 7.0f));
    assert(float_equals(a->data[2], 9.0f));

    vector_free(a);
    vector_free(b);
    TEST_PASSED;
}

void test_vector_scale_by_zero() {
    Vector *v = vector_ones(3);
    Vector *result = vector_create(3);

    vector_scale(result, v, 0.0f);

    assert(float_equals(result->data[0], 0.0f));
    assert(float_equals(result->data[1], 0.0f));
    assert(float_equals(result->data[2], 0.0f));

    vector_free(v);
    vector_free(result);
    TEST_PASSED;
}

void test_vector_scale_by_negative() {
    Vector *v = vector_create(3);
    Vector *result = vector_create(3);

    v->data[0] = 1.0f;
    v->data[1] = 2.0f;
    v->data[2] = 3.0f;

    vector_scale(result, v, -2.0f);

    assert(float_equals(result->data[0], -2.0f));
    assert(float_equals(result->data[1], -4.0f));
    assert(float_equals(result->data[2], -6.0f));

    vector_free(v);
    vector_free(result);
    TEST_PASSED;
}

// =============================================================================
// Test Runner
// =============================================================================

void run_vector_tests(void) {
    printf("\n=== Vector Creation and Destruction Tests ===\n");
    test_vector_create();
    test_vector_zeros();
    test_vector_ones();
    test_vector_random();

    printf("\n=== Vector Basic Operations Tests ===\n");
    test_vector_add();
    test_vector_subtract();
    test_vector_scale();
    test_vector_dot();
    test_vector_dot_orthogonal();

    printf("\n=== Vector Element-wise Operations Tests ===\n");
    test_vector_multiply();
    test_vector_divide();

    printf("\n=== Vector Utility Functions Tests ===\n");
    test_vector_copy();
    test_vector_sum();
    test_vector_sum_negative();
    test_vector_min();
    test_vector_max();
    test_vector_print();

    printf("\n=== Vector Other Functions Tests ===\n");
    test_vector_outer_product();

    printf("\n=== Vector Edge Cases Tests ===\n");
    test_vector_single_element();
    test_vector_add_in_place();
    test_vector_scale_by_zero();
    test_vector_scale_by_negative();

    printf("\n=== All Vector Tests Passed! ===\n");
}
