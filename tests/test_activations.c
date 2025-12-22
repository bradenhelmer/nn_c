/*
 * test_activations.c - Comprehensive tests for activation functions
 */

#include "../src/activations/activations.h"
#include "test_runner.h"
#include "utils/utils.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

// =============================================================================
// Singular Float Activation Function Tests
// =============================================================================

void test_singular_sigmoid_basic() {
    // Test sigmoid at key points
    assert(float_equals(sigmoid_scalar(0.0f), 0.5f));
    assert(float_equals(sigmoid_scalar(1.0f), 1.0f / (1.0f + expf(-1.0f))));
    assert(float_equals(sigmoid_scalar(-1.0f), 1.0f / (1.0f + expf(1.0f))));
    TEST_PASSED;
}

void test_singular_sigmoid_extremes() {
    // Test large positive values - should approach 1
    assert(sigmoid_scalar(10.0f) > 0.9999f);
    assert(sigmoid_scalar(100.0f) > 0.9999f);

    // Test large negative values - should approach 0
    assert(sigmoid_scalar(-10.0f) < 0.0001f);
    assert(sigmoid_scalar(-100.0f) < 0.0001f);
    TEST_PASSED;
}

void test_singular_sigmoid_derivative_basic() {
    // sigmoid'(x) = sigmoid(x) * (1 - sigmoid(x))
    float s1 = 0.5f;    // sigmoid(0)
    float s2 = 0.7311f; // sigmoid(1)
    float s3 = 0.2689f; // sigmoid(-1)

    assert(float_equals(sigmoid_scalar_derivative(s1), 0.25f));
    assert(float_equals(sigmoid_scalar_derivative(s2), s2 * (1.0f - s2)));
    assert(float_equals(sigmoid_scalar_derivative(s3), s3 * (1.0f - s3)));
    TEST_PASSED;
}

void test_singular_sigmoid_derivative_extremes() {
    // At extremes, derivative should be close to 0
    assert(float_equals(sigmoid_scalar_derivative(0.0f), 0.0f));
    assert(float_equals(sigmoid_scalar_derivative(1.0f), 0.0f));
    TEST_PASSED;
}

void test_singular_relu_basic() {
    // Test ReLU at key points
    assert(float_equals(relu_scalar(0.0f), 0.0f));
    assert(float_equals(relu_scalar(1.0f), 1.0f));
    assert(float_equals(relu_scalar(-1.0f), 0.0f));
    assert(float_equals(relu_scalar(5.5f), 5.5f));
    assert(float_equals(relu_scalar(-10.0f), 0.0f));
    TEST_PASSED;
}

void test_singular_relu_derivative_basic() {
    // ReLU derivative: 1 for x >= 0, 0 for x < 0
    assert(float_equals(relu_scalar_derivative(0.0f), 1.0f));
    assert(float_equals(relu_scalar_derivative(1.0f), 1.0f));
    assert(float_equals(relu_scalar_derivative(-1.0f), 0.0f));
    assert(float_equals(relu_scalar_derivative(5.5f), 1.0f));
    assert(float_equals(relu_scalar_derivative(-10.0f), 0.0f));
    TEST_PASSED;
}

void test_singular_tanh_derivative_basic() {
    // tanh'(x) = 1 - tanh(x)^2
    float t1 = 0.0f;     // tanh(0)
    float t2 = 0.7616f;  // tanh(1)
    float t3 = -0.7616f; // tanh(-1)

    assert(float_equals(tanh_scalar_derivative(t1), 1.0f));
    assert(float_equals(tanh_scalar_derivative(t2), 1.0f - (t2 * t2)));
    assert(float_equals(tanh_scalar_derivative(t3), 1.0f - (t3 * t3)));
    TEST_PASSED;
}

void test_singular_tanh_derivative_extremes() {
    // At extremes, derivative should be close to 0
    assert(float_equals(tanh_scalar_derivative(1.0f), 0.0f));
    assert(float_equals(tanh_scalar_derivative(-1.0f), 0.0f));
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

    printf("\n=== All Activation Tests Passed! ===\n");
}
