/*
 * test_activations.c - Comprehensive tests for activation functions
 */

#include "../src/activations/activations.h"
#include "../src/linalg/matrix.h"
#include "../src/linalg/vector.h"
#include "test_runner.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// =============================================================================
// Singular Float Activation Function Tests
// =============================================================================

void test_singular_sigmoid_basic() {
    // Test sigmoid at key points
    assert(float_equals(sigmoid(0.0f), 0.5f));
    assert(float_equals(sigmoid(1.0f), 1.0f / (1.0f + expf(-1.0f))));
    assert(float_equals(sigmoid(-1.0f), 1.0f / (1.0f + expf(1.0f))));
    TEST_PASSED;
}

void test_singular_sigmoid_extremes() {
    // Test large positive values - should approach 1
    assert(sigmoid(10.0f) > 0.9999f);
    assert(sigmoid(100.0f) > 0.9999f);

    // Test large negative values - should approach 0
    assert(sigmoid(-10.0f) < 0.0001f);
    assert(sigmoid(-100.0f) < 0.0001f);
    TEST_PASSED;
}

void test_singular_sigmoid_derivative_basic() {
    // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    float s1 = 0.5f;   // sigmoid(0)
    float s2 = 0.7311f; // sigmoid(1)
    float s3 = 0.2689f; // sigmoid(-1)

    assert(float_equals(sigmoid_derivative(s1), 0.25f));
    assert(float_equals(sigmoid_derivative(s2), s2 * (1.0f - s2)));
    assert(float_equals(sigmoid_derivative(s3), s3 * (1.0f - s3)));
    TEST_PASSED;
}

void test_singular_sigmoid_derivative_extremes() {
    // At extremes, derivative should be close to 0
    assert(float_equals(sigmoid_derivative(0.0f), 0.0f));
    assert(float_equals(sigmoid_derivative(1.0f), 0.0f));
    TEST_PASSED;
}

void test_singular_relu_basic() {
    // Test ReLU at key points
    assert(float_equals(relu(0.0f), 0.0f));
    assert(float_equals(relu(1.0f), 1.0f));
    assert(float_equals(relu(-1.0f), 0.0f));
    assert(float_equals(relu(5.5f), 5.5f));
    assert(float_equals(relu(-10.0f), 0.0f));
    TEST_PASSED;
}

void test_singular_relu_derivative_basic() {
    // ReLU derivative: 1 for x >= 0, 0 for x < 0
    assert(float_equals(relu_derivative(0.0f), 1.0f));
    assert(float_equals(relu_derivative(1.0f), 1.0f));
    assert(float_equals(relu_derivative(-1.0f), 0.0f));
    assert(float_equals(relu_derivative(5.5f), 1.0f));
    assert(float_equals(relu_derivative(-10.0f), 0.0f));
    TEST_PASSED;
}

void test_singular_tanh_derivative_basic() {
    // tanh'(x) = 1 - tanh(x)^2
    float t1 = 0.0f;     // tanh(0)
    float t2 = 0.7616f;  // tanh(1)
    float t3 = -0.7616f; // tanh(-1)

    assert(float_equals(tanh_derivative(t1), 1.0f));
    assert(float_equals(tanh_derivative(t2), 1.0f - (t2 * t2)));
    assert(float_equals(tanh_derivative(t3), 1.0f - (t3 * t3)));
    TEST_PASSED;
}

void test_singular_tanh_derivative_extremes() {
    // At extremes, derivative should be close to 0
    assert(float_equals(tanh_derivative(1.0f), 0.0f));
    assert(float_equals(tanh_derivative(-1.0f), 0.0f));
    TEST_PASSED;
}

// =============================================================================
// Vector Sigmoid Tests
// =============================================================================

void test_sigmoid_basic() {
    Vector *input = vector_create(3);
    Vector *result = vector_create(3);

    input->data[0] = 0.0f;
    input->data[1] = 1.0f;
    input->data[2] = -1.0f;

    vector_sigmoid(result, input);

    // sigmoid(0) = 0.5
    assert(float_equals(result->data[0], 0.5f));
    // sigmoid(1) ≈ 0.7311
    assert(float_equals(result->data[1], 1.0f / (1.0f + expf(-1.0f))));
    // sigmoid(-1) ≈ 0.2689
    assert(float_equals(result->data[2], 1.0f / (1.0f + expf(1.0f))));

    vector_free(input);
    vector_free(result);
    TEST_PASSED;
}

void test_sigmoid_large_positive() {
    Vector *input = vector_create(2);
    Vector *result = vector_create(2);

    input->data[0] = 10.0f;
    input->data[1] = 100.0f;

    vector_sigmoid(result, input);

    // For large positive values, sigmoid approaches 1
    assert(result->data[0] > 0.9999f);
    assert(result->data[1] > 0.9999f);

    vector_free(input);
    vector_free(result);
    TEST_PASSED;
}

void test_sigmoid_large_negative() {
    Vector *input = vector_create(2);
    Vector *result = vector_create(2);

    input->data[0] = -10.0f;
    input->data[1] = -100.0f;

    vector_sigmoid(result, input);

    // For large negative values, sigmoid approaches 0
    assert(result->data[0] < 0.0001f);
    assert(result->data[1] < 0.0001f);

    vector_free(input);
    vector_free(result);
    TEST_PASSED;
}

void test_sigmoid_derivative_basic() {
    Vector *sigmoid_output = vector_create(3);
    Vector *result = vector_create(3);

    // Test at key points
    sigmoid_output->data[0] = 0.5f;    // sigmoid'(0) = 0.25
    sigmoid_output->data[1] = 0.7311f; // sigmoid(1)
    sigmoid_output->data[2] = 0.2689f; // sigmoid(-1)

    vector_sigmoid_derivative(result, sigmoid_output);

    // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    assert(float_equals(result->data[0], 0.25f));
    assert(float_equals(result->data[1], 0.7311f * (1.0f - 0.7311f)));
    assert(float_equals(result->data[2], 0.2689f * (1.0f - 0.2689f)));

    vector_free(sigmoid_output);
    vector_free(result);
    TEST_PASSED;
}

void test_sigmoid_derivative_extremes() {
    Vector *sigmoid_output = vector_create(2);
    Vector *result = vector_create(2);

    sigmoid_output->data[0] = 0.0f;
    sigmoid_output->data[1] = 1.0f;

    vector_sigmoid_derivative(result, sigmoid_output);

    // At extremes, derivative should be close to 0
    assert(float_equals(result->data[0], 0.0f));
    assert(float_equals(result->data[1], 0.0f));

    vector_free(sigmoid_output);
    vector_free(result);
    TEST_PASSED;
}

// =============================================================================
// Vector ReLU Tests
// =============================================================================

void test_relu_basic() {
    Vector *input = vector_create(5);
    Vector *result = vector_create(5);

    input->data[0] = 0.0f;
    input->data[1] = 1.0f;
    input->data[2] = -1.0f;
    input->data[3] = 5.5f;
    input->data[4] = -10.0f;

    vector_relu(result, input);

    assert(float_equals(result->data[0], 0.0f));
    assert(float_equals(result->data[1], 1.0f));
    assert(float_equals(result->data[2], 0.0f));
    assert(float_equals(result->data[3], 5.5f));
    assert(float_equals(result->data[4], 0.0f));

    vector_free(input);
    vector_free(result);
    TEST_PASSED;
}

void test_relu_all_positive() {
    Vector *input = vector_create(3);
    Vector *result = vector_create(3);

    input->data[0] = 1.0f;
    input->data[1] = 2.0f;
    input->data[2] = 3.0f;

    vector_relu(result, input);

    assert(float_equals(result->data[0], 1.0f));
    assert(float_equals(result->data[1], 2.0f));
    assert(float_equals(result->data[2], 3.0f));

    vector_free(input);
    vector_free(result);
    TEST_PASSED;
}

void test_relu_all_negative() {
    Vector *input = vector_create(3);
    Vector *result = vector_create(3);

    input->data[0] = -1.0f;
    input->data[1] = -2.0f;
    input->data[2] = -3.0f;

    vector_relu(result, input);

    assert(float_equals(result->data[0], 0.0f));
    assert(float_equals(result->data[1], 0.0f));
    assert(float_equals(result->data[2], 0.0f));

    vector_free(input);
    vector_free(result);
    TEST_PASSED;
}

void test_relu_derivative_basic() {
    Vector *input = vector_create(5);
    Vector *result = vector_create(5);

    input->data[0] = 0.0f;
    input->data[1] = 1.0f;
    input->data[2] = -1.0f;
    input->data[3] = 5.5f;
    input->data[4] = -10.0f;

    vector_relu_derivative(result, input);

    // ReLU derivative is 0 for x < 0, 1 for x >= 0
    // Note: at x=0, the implementation uses x < 0, so derivative is 1
    assert(float_equals(result->data[0], 1.0f));
    assert(float_equals(result->data[1], 1.0f));
    assert(float_equals(result->data[2], 0.0f));
    assert(float_equals(result->data[3], 1.0f));
    assert(float_equals(result->data[4], 0.0f));

    vector_free(input);
    vector_free(result);
    TEST_PASSED;
}

// =============================================================================
// Vector Tanh Tests
// =============================================================================

void test_tanh_basic() {
    Vector *input = vector_create(3);
    Vector *result = vector_create(3);

    input->data[0] = 0.0f;
    input->data[1] = 1.0f;
    input->data[2] = -1.0f;

    vector_tanh_activation(result, input);

    // tanh(0) = 0
    assert(float_equals(result->data[0], 0.0f));
    // tanh(1) ≈ 0.7616
    assert(float_equals(result->data[1], tanhf(1.0f)));
    // tanh(-1) ≈ -0.7616
    assert(float_equals(result->data[2], tanhf(-1.0f)));

    vector_free(input);
    vector_free(result);
    TEST_PASSED;
}

void test_tanh_large_values() {
    Vector *input = vector_create(4);
    Vector *result = vector_create(4);

    input->data[0] = 10.0f;
    input->data[1] = -10.0f;
    input->data[2] = 100.0f;
    input->data[3] = -100.0f;

    vector_tanh_activation(result, input);

    // For large positive values, tanh approaches 1
    assert(result->data[0] > 0.9999f);
    assert(result->data[2] > 0.9999f);
    // For large negative values, tanh approaches -1
    assert(result->data[1] < -0.9999f);
    assert(result->data[3] < -0.9999f);

    vector_free(input);
    vector_free(result);
    TEST_PASSED;
}

void test_tanh_derivative_basic() {
    Vector *tanh_output = vector_create(3);
    Vector *result = vector_create(3);

    tanh_output->data[0] = 0.0f;     // tanh(0)
    tanh_output->data[1] = 0.7616f;  // tanh(1)
    tanh_output->data[2] = -0.7616f; // tanh(-1)

    vector_tanh_derivative(result, tanh_output);

    // tanh'(x) = 1 - tanh(x)^2
    assert(float_equals(result->data[0], 1.0f));
    assert(float_equals(result->data[1], 1.0f - (0.7616f * 0.7616f)));
    assert(float_equals(result->data[2], 1.0f - (0.7616f * 0.7616f)));

    vector_free(tanh_output);
    vector_free(result);
    TEST_PASSED;
}

void test_tanh_derivative_extremes() {
    Vector *tanh_output = vector_create(2);
    Vector *result = vector_create(2);

    tanh_output->data[0] = 1.0f;
    tanh_output->data[1] = -1.0f;

    vector_tanh_derivative(result, tanh_output);

    // At extremes, derivative should be close to 0
    assert(float_equals(result->data[0], 0.0f));
    assert(float_equals(result->data[1], 0.0f));

    vector_free(tanh_output);
    vector_free(result);
    TEST_PASSED;
}

// =============================================================================
// Vector Softmax Tests
// =============================================================================

void test_softmax_basic() {
    Vector *input = vector_create(3);
    Vector *result = vector_create(3);

    input->data[0] = 1.0f;
    input->data[1] = 2.0f;
    input->data[2] = 3.0f;

    vector_softmax(result, input);

    // Check that all values are positive
    assert(result->data[0] > 0.0f);
    assert(result->data[1] > 0.0f);
    assert(result->data[2] > 0.0f);

    // Check that they sum to 1
    float sum = result->data[0] + result->data[1] + result->data[2];
    assert(float_equals(sum, 1.0f));

    // Check relative ordering is preserved (larger input -> larger output)
    assert(result->data[2] > result->data[1]);
    assert(result->data[1] > result->data[0]);

    vector_free(input);
    vector_free(result);
    TEST_PASSED;
}

void test_softmax_uniform() {
    Vector *input = vector_create(4);
    Vector *result = vector_create(4);

    // All same values
    for (int i = 0; i < 4; i++) {
        input->data[i] = 5.0f;
    }

    vector_softmax(result, input);

    // All outputs should be equal (1/n)
    for (int i = 0; i < 4; i++) {
        assert(float_equals(result->data[i], 0.25f));
    }

    // Should sum to 1
    float sum = vector_sum(result);
    assert(float_equals(sum, 1.0f));

    vector_free(input);
    vector_free(result);
    TEST_PASSED;
}

void test_softmax_large_values() {
    Vector *input = vector_create(3);
    Vector *result = vector_create(3);

    // Test numerical stability with large values
    input->data[0] = 1000.0f;
    input->data[1] = 1001.0f;
    input->data[2] = 1002.0f;

    vector_softmax(result, input);

    // Should still sum to 1 (tests numerical stability)
    float sum = result->data[0] + result->data[1] + result->data[2];
    assert(float_equals(sum, 1.0f));

    // All values should be positive and finite
    for (int i = 0; i < 3; i++) {
        assert(result->data[i] > 0.0f);
        assert(isfinite(result->data[i]));
    }

    vector_free(input);
    vector_free(result);
    TEST_PASSED;
}

void test_softmax_with_negatives() {
    Vector *input = vector_create(3);
    Vector *result = vector_create(3);

    input->data[0] = -1.0f;
    input->data[1] = 0.0f;
    input->data[2] = 1.0f;

    vector_softmax(result, input);

    // Should sum to 1
    float sum = vector_sum(result);
    assert(float_equals(sum, 1.0f));

    // All values should be positive
    assert(result->data[0] > 0.0f);
    assert(result->data[1] > 0.0f);
    assert(result->data[2] > 0.0f);

    // Check ordering
    assert(result->data[2] > result->data[1]);
    assert(result->data[1] > result->data[0]);

    vector_free(input);
    vector_free(result);
    TEST_PASSED;
}

// =============================================================================
// Matrix Sigmoid Tests
// =============================================================================

void test_matrix_sigmoid_basic() {
    Matrix *input = matrix_create(2, 2);
    Matrix *result = matrix_create(2, 2);

    matrix_set(input, 0, 0, 0.0f);
    matrix_set(input, 0, 1, 1.0f);
    matrix_set(input, 1, 0, -1.0f);
    matrix_set(input, 1, 1, 2.0f);

    matrix_sigmoid(result, input);

    // sigmoid(0) = 0.5
    assert(float_equals(matrix_get(result, 0, 0), 0.5f));
    // sigmoid(1) ≈ 0.7311
    assert(float_equals(matrix_get(result, 0, 1), 1.0f / (1.0f + expf(-1.0f))));
    // sigmoid(-1) ≈ 0.2689
    assert(float_equals(matrix_get(result, 1, 0), 1.0f / (1.0f + expf(1.0f))));
    // sigmoid(2) ≈ 0.8808
    assert(float_equals(matrix_get(result, 1, 1), 1.0f / (1.0f + expf(-2.0f))));

    matrix_free(input);
    matrix_free(result);
    TEST_PASSED;
}

void test_matrix_sigmoid_derivative_basic() {
    Matrix *sigmoid_output = matrix_create(2, 2);
    Matrix *result = matrix_create(2, 2);

    matrix_set(sigmoid_output, 0, 0, 0.5f);
    matrix_set(sigmoid_output, 0, 1, 0.7311f);
    matrix_set(sigmoid_output, 1, 0, 0.2689f);
    matrix_set(sigmoid_output, 1, 1, 0.8808f);

    matrix_sigmoid_derivative(result, sigmoid_output);

    // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    assert(float_equals(matrix_get(result, 0, 0), 0.25f));
    assert(float_equals(matrix_get(result, 0, 1), 0.7311f * (1.0f - 0.7311f)));
    assert(float_equals(matrix_get(result, 1, 0), 0.2689f * (1.0f - 0.2689f)));
    assert(float_equals(matrix_get(result, 1, 1), 0.8808f * (1.0f - 0.8808f)));

    matrix_free(sigmoid_output);
    matrix_free(result);
    TEST_PASSED;
}

// =============================================================================
// Matrix ReLU Tests
// =============================================================================

void test_matrix_relu_basic() {
    Matrix *input = matrix_create(2, 3);
    Matrix *result = matrix_create(2, 3);

    matrix_set(input, 0, 0, 1.0f);
    matrix_set(input, 0, 1, -2.0f);
    matrix_set(input, 0, 2, 3.0f);
    matrix_set(input, 1, 0, -4.0f);
    matrix_set(input, 1, 1, 5.0f);
    matrix_set(input, 1, 2, 0.0f);

    matrix_relu(result, input);

    assert(float_equals(matrix_get(result, 0, 0), 1.0f));
    assert(float_equals(matrix_get(result, 0, 1), 0.0f));
    assert(float_equals(matrix_get(result, 0, 2), 3.0f));
    assert(float_equals(matrix_get(result, 1, 0), 0.0f));
    assert(float_equals(matrix_get(result, 1, 1), 5.0f));
    assert(float_equals(matrix_get(result, 1, 2), 0.0f));

    matrix_free(input);
    matrix_free(result);
    TEST_PASSED;
}

void test_matrix_relu_derivative_basic() {
    Matrix *input = matrix_create(2, 3);
    Matrix *result = matrix_create(2, 3);

    matrix_set(input, 0, 0, 1.0f);
    matrix_set(input, 0, 1, -2.0f);
    matrix_set(input, 0, 2, 3.0f);
    matrix_set(input, 1, 0, -4.0f);
    matrix_set(input, 1, 1, 5.0f);
    matrix_set(input, 1, 2, 0.0f);

    matrix_relu_derivative(result, input);

    // Derivative is 1 for x >= 0, 0 for x < 0
    assert(float_equals(matrix_get(result, 0, 0), 1.0f));
    assert(float_equals(matrix_get(result, 0, 1), 0.0f));
    assert(float_equals(matrix_get(result, 0, 2), 1.0f));
    assert(float_equals(matrix_get(result, 1, 0), 0.0f));
    assert(float_equals(matrix_get(result, 1, 1), 1.0f));
    assert(float_equals(matrix_get(result, 1, 2), 1.0f));

    matrix_free(input);
    matrix_free(result);
    TEST_PASSED;
}

// =============================================================================
// Edge Cases Tests
// =============================================================================

void test_sigmoid_in_place() {
    Vector *v = vector_create(3);
    v->data[0] = 0.0f;
    v->data[1] = 1.0f;
    v->data[2] = -1.0f;

    // Use same vector for input and output
    vector_sigmoid(v, v);

    assert(float_equals(v->data[0], 0.5f));
    assert(float_equals(v->data[1], 1.0f / (1.0f + expf(-1.0f))));
    assert(float_equals(v->data[2], 1.0f / (1.0f + expf(1.0f))));

    vector_free(v);
    TEST_PASSED;
}

void test_relu_in_place() {
    Vector *v = vector_create(3);
    v->data[0] = 1.0f;
    v->data[1] = -2.0f;
    v->data[2] = 3.0f;

    // Use same vector for input and output
    vector_relu(v, v);

    assert(float_equals(v->data[0], 1.0f));
    assert(float_equals(v->data[1], 0.0f));
    assert(float_equals(v->data[2], 3.0f));

    vector_free(v);
    TEST_PASSED;
}

void test_tanh_in_place() {
    Vector *v = vector_create(2);
    v->data[0] = 0.0f;
    v->data[1] = 1.0f;

    // Use same vector for input and output
    vector_tanh_activation(v, v);

    assert(float_equals(v->data[0], 0.0f));
    assert(float_equals(v->data[1], tanhf(1.0f)));

    vector_free(v);
    TEST_PASSED;
}

void test_softmax_single_element() {
    Vector *input = vector_create(1);
    Vector *result = vector_create(1);

    input->data[0] = 5.0f;

    vector_softmax(result, input);

    // Single element softmax should always be 1
    assert(float_equals(result->data[0], 1.0f));

    vector_free(input);
    vector_free(result);
    TEST_PASSED;
}

void test_activation_chain() {
    // Test chaining multiple activations
    Vector *v = vector_create(3);
    Vector *temp1 = vector_create(3);
    Vector *temp2 = vector_create(3);

    v->data[0] = -2.0f;
    v->data[1] = 0.0f;
    v->data[2] = 2.0f;

    // Apply ReLU first
    vector_relu(temp1, v);

    // Then apply sigmoid
    vector_sigmoid(temp2, temp1);

    // After ReLU: [0, 0, 2]
    // After sigmoid: [0.5, 0.5, sigmoid(2)]
    assert(float_equals(temp2->data[0], 0.5f));
    assert(float_equals(temp2->data[1], 0.5f));
    assert(float_equals(temp2->data[2], 1.0f / (1.0f + expf(-2.0f))));

    vector_free(v);
    vector_free(temp1);
    vector_free(temp2);
    TEST_PASSED;
}

// =============================================================================
// Test Runner
// =============================================================================

void run_activations_tests(void) {
    printf("\n=== Singular Float Activation Tests ===\n");
    test_singular_sigmoid_basic();
    test_singular_sigmoid_extremes();
    test_singular_sigmoid_derivative_basic();
    test_singular_sigmoid_derivative_extremes();
    test_singular_relu_basic();
    test_singular_relu_derivative_basic();
    test_singular_tanh_derivative_basic();
    test_singular_tanh_derivative_extremes();

    printf("\n=== Vector Sigmoid Tests ===\n");
    test_sigmoid_basic();
    test_sigmoid_large_positive();
    test_sigmoid_large_negative();
    test_sigmoid_derivative_basic();
    test_sigmoid_derivative_extremes();

    printf("\n=== Vector ReLU Tests ===\n");
    test_relu_basic();
    test_relu_all_positive();
    test_relu_all_negative();
    test_relu_derivative_basic();

    printf("\n=== Vector Tanh Tests ===\n");
    test_tanh_basic();
    test_tanh_large_values();
    test_tanh_derivative_basic();
    test_tanh_derivative_extremes();

    printf("\n=== Vector Softmax Tests ===\n");
    test_softmax_basic();
    test_softmax_uniform();
    test_softmax_large_values();
    test_softmax_with_negatives();

    printf("\n=== Matrix Sigmoid Tests ===\n");
    test_matrix_sigmoid_basic();
    test_matrix_sigmoid_derivative_basic();

    printf("\n=== Matrix ReLU Tests ===\n");
    test_matrix_relu_basic();
    test_matrix_relu_derivative_basic();

    printf("\n=== Activation Edge Cases Tests ===\n");
    test_sigmoid_in_place();
    test_relu_in_place();
    test_tanh_in_place();
    test_softmax_single_element();
    test_activation_chain();

    printf("\n=== All Activation Tests Passed! ===\n");
}
