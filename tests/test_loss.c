/*
 * test_loss.c - Comprehensive tests for loss functions
 */

#include "../src/linalg/vector.h"
#include "../src/nn/loss.h"
#include "test_runner.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// =============================================================================
// Singular MSE Loss Tests
// =============================================================================

void test_mse_loss_zero_error() {
    // When prediction equals target, loss should be 0
    float result = mse_loss(5.0f, 5.0f);
    assert(float_equals(result, 0.0f));
    TEST_PASSED;
}

void test_mse_loss_basic() {
    // MSE = 0.5 * (predicted - target)^2
    // For predicted=3, target=1: 0.5 * (3-1)^2 = 0.5 * 4 = 2.0
    float result = mse_loss(3.0f, 1.0f);
    assert(float_equals(result, 2.0f));
    TEST_PASSED;
}

void test_mse_loss_negative_error() {
    // For predicted=1, target=3: 0.5 * (1-3)^2 = 0.5 * 4 = 2.0
    // Should be same as positive error
    float result = mse_loss(1.0f, 3.0f);
    assert(float_equals(result, 2.0f));
    TEST_PASSED;
}

void test_mse_loss_fractional() {
    // For predicted=2.5, target=2.0: 0.5 * (0.5)^2 = 0.125
    float result = mse_loss(2.5f, 2.0f);
    assert(float_equals(result, 0.125f));
    TEST_PASSED;
}

// =============================================================================
// Vector MSE Loss Tests
// =============================================================================

void test_vector_mse_zero_error() {
    Vector *prediction = vector_create(3);
    Vector *target = vector_create(3);

    prediction->data[0] = 1.0f;
    prediction->data[1] = 2.0f;
    prediction->data[2] = 3.0f;

    target->data[0] = 1.0f;
    target->data[1] = 2.0f;
    target->data[2] = 3.0f;

    float loss = vector_mse(prediction, target);
    assert(float_equals(loss, 0.0f));

    vector_free(prediction);
    vector_free(target);
    TEST_PASSED;
}

void test_vector_mse_basic() {
    Vector *prediction = vector_create(3);
    Vector *target = vector_create(3);

    prediction->data[0] = 1.0f;
    prediction->data[1] = 2.0f;
    prediction->data[2] = 3.0f;

    target->data[0] = 0.0f;
    target->data[1] = 0.0f;
    target->data[2] = 0.0f;

    // MSE = mean((pred - target)^2)
    // = (1^2 + 2^2 + 3^2) / 3 = (1 + 4 + 9) / 3 = 14/3 ≈ 4.6667
    float loss = vector_mse(prediction, target);
    assert(float_equals(loss, 14.0f / 3.0f));

    vector_free(prediction);
    vector_free(target);
    TEST_PASSED;
}

void test_vector_mse_mixed_errors() {
    Vector *prediction = vector_create(4);
    Vector *target = vector_create(4);

    prediction->data[0] = 2.0f;
    prediction->data[1] = -1.0f;
    prediction->data[2] = 3.0f;
    prediction->data[3] = 0.0f;

    target->data[0] = 1.0f;
    target->data[1] = 1.0f;
    target->data[2] = 2.0f;
    target->data[3] = 0.5f;

    // Errors: [1, -2, 1, -0.5]
    // MSE = (1 + 4 + 1 + 0.25) / 4 = 6.25 / 4 = 1.5625
    float loss = vector_mse(prediction, target);
    assert(float_equals(loss, 1.5625f));

    vector_free(prediction);
    vector_free(target);
    TEST_PASSED;
}

void test_vector_mse_single_element() {
    Vector *prediction = vector_create(1);
    Vector *target = vector_create(1);

    prediction->data[0] = 3.0f;
    target->data[0] = 1.0f;

    // MSE = (3-1)^2 / 1 = 4
    float loss = vector_mse(prediction, target);
    assert(float_equals(loss, 4.0f));

    vector_free(prediction);
    vector_free(target);
    TEST_PASSED;
}

// =============================================================================
// Vector MSE Derivative Tests
// =============================================================================

void test_vector_mse_derivative_zero_error() {
    Vector *prediction = vector_create(3);
    Vector *target = vector_create(3);
    Vector *result = vector_create(3);

    prediction->data[0] = 1.0f;
    prediction->data[1] = 2.0f;
    prediction->data[2] = 3.0f;

    target->data[0] = 1.0f;
    target->data[1] = 2.0f;
    target->data[2] = 3.0f;

    vector_mse_derivative(result, prediction, target);

    // When prediction equals target, derivative should be 0
    assert(float_equals(result->data[0], 0.0f));
    assert(float_equals(result->data[1], 0.0f));
    assert(float_equals(result->data[2], 0.0f));

    vector_free(prediction);
    vector_free(target);
    vector_free(result);
    TEST_PASSED;
}

void test_vector_mse_derivative_basic() {
    Vector *prediction = vector_create(3);
    Vector *target = vector_create(3);
    Vector *result = vector_create(3);

    prediction->data[0] = 3.0f;
    prediction->data[1] = 2.0f;
    prediction->data[2] = 1.0f;

    target->data[0] = 0.0f;
    target->data[1] = 0.0f;
    target->data[2] = 0.0f;

    vector_mse_derivative(result, prediction, target);

    // d/dx MSE = (prediction - target) / n
    // For n=3: [3/3, 2/3, 1/3] = [1.0, 0.6667, 0.3333]
    assert(float_equals(result->data[0], 1.0f));
    assert(float_equals(result->data[1], 2.0f / 3.0f));
    assert(float_equals(result->data[2], 1.0f / 3.0f));

    vector_free(prediction);
    vector_free(target);
    vector_free(result);
    TEST_PASSED;
}

void test_vector_mse_derivative_negative_errors() {
    Vector *prediction = vector_create(3);
    Vector *target = vector_create(3);
    Vector *result = vector_create(3);

    prediction->data[0] = 1.0f;
    prediction->data[1] = 2.0f;
    prediction->data[2] = 3.0f;

    target->data[0] = 4.0f;
    target->data[1] = 5.0f;
    target->data[2] = 6.0f;

    vector_mse_derivative(result, prediction, target);

    // Errors: [-3, -3, -3]
    // Derivatives: [-1.0, -1.0, -1.0]
    assert(float_equals(result->data[0], -1.0f));
    assert(float_equals(result->data[1], -1.0f));
    assert(float_equals(result->data[2], -1.0f));

    vector_free(prediction);
    vector_free(target);
    vector_free(result);
    TEST_PASSED;
}

void test_vector_mse_derivative_in_place() {
    Vector *prediction = vector_create(2);
    Vector *target = vector_create(2);

    prediction->data[0] = 4.0f;
    prediction->data[1] = 2.0f;

    target->data[0] = 0.0f;
    target->data[1] = 0.0f;

    // Use prediction as both input and output
    vector_mse_derivative(prediction, prediction, target);

    assert(float_equals(prediction->data[0], 2.0f));
    assert(float_equals(prediction->data[1], 1.0f));

    vector_free(prediction);
    vector_free(target);
    TEST_PASSED;
}

// =============================================================================
// Vector Cross-Entropy Loss Tests
// =============================================================================

void test_vector_cross_entropy_basic() {
    Vector *prediction = vector_create(3);
    Vector *target = vector_create(3);

    // One-hot encoding: target is class 1
    target->data[0] = 0.0f;
    target->data[1] = 1.0f;
    target->data[2] = 0.0f;

    // Prediction probabilities (should sum to ~1)
    prediction->data[0] = 0.1f;
    prediction->data[1] = 0.8f;
    prediction->data[2] = 0.1f;

    float loss = vector_cross_entropy(prediction, target);

    // CE = -sum(target * log(prediction))
    // = -(0*log(0.1) + 1*log(0.8) + 0*log(0.1))
    // = -log(0.8) ≈ 0.2231
    float expected = -logf(0.8f);
    assert(float_equals(loss, expected));

    vector_free(prediction);
    vector_free(target);
    TEST_PASSED;
}

void test_vector_cross_entropy_perfect_prediction() {
    Vector *prediction = vector_create(3);
    Vector *target = vector_create(3);

    // One-hot encoding: target is class 2
    target->data[0] = 0.0f;
    target->data[1] = 0.0f;
    target->data[2] = 1.0f;

    // Perfect prediction (close to 1.0 to avoid log(1) issues)
    prediction->data[0] = 0.0f;
    prediction->data[1] = 0.0f;
    prediction->data[2] = 0.9999f;

    float loss = vector_cross_entropy(prediction, target);

    // CE = -log(0.9999) ≈ 0.0001
    // Should be very close to 0
    assert(loss < 0.001f);

    vector_free(prediction);
    vector_free(target);
    TEST_PASSED;
}

void test_vector_cross_entropy_multi_class() {
    Vector *prediction = vector_create(4);
    Vector *target = vector_create(4);

    // Soft targets (not one-hot)
    target->data[0] = 0.7f;
    target->data[1] = 0.2f;
    target->data[2] = 0.1f;
    target->data[3] = 0.0f;

    prediction->data[0] = 0.6f;
    prediction->data[1] = 0.3f;
    prediction->data[2] = 0.05f;
    prediction->data[3] = 0.05f;

    float loss = vector_cross_entropy(prediction, target);

    // CE = -(0.7*log(0.6) + 0.2*log(0.3) + 0.1*log(0.05) + 0*log(0.05))
    float expected = -(0.7f * logf(0.6f) + 0.2f * logf(0.3f) + 0.1f * logf(0.05f));
    assert(float_equals(loss, expected));

    vector_free(prediction);
    vector_free(target);
    TEST_PASSED;
}

void test_vector_cross_entropy_epsilon_protection() {
    Vector *prediction = vector_create(2);
    Vector *target = vector_create(2);

    target->data[0] = 1.0f;
    target->data[1] = 0.0f;

    // Very small prediction (tests epsilon protection)
    prediction->data[0] = 0.0f;
    prediction->data[1] = 1.0f;

    // Should not crash or return inf due to EPSILON
    float loss = vector_cross_entropy(prediction, target);
    assert(isfinite(loss));

    vector_free(prediction);
    vector_free(target);
    TEST_PASSED;
}

// =============================================================================
// Vector Cross-Entropy Derivative Tests
// =============================================================================

void test_vector_cross_entropy_derivative_basic() {
    Vector *prediction = vector_create(3);
    Vector *target = vector_create(3);
    Vector *result = vector_create(3);

    target->data[0] = 0.0f;
    target->data[1] = 1.0f;
    target->data[2] = 0.0f;

    prediction->data[0] = 0.1f;
    prediction->data[1] = 0.8f;
    prediction->data[2] = 0.1f;

    vector_cross_entropy_derivative(result, prediction, target);

    // d/dx CE = -target / prediction
    assert(float_equals(result->data[0], 0.0f));         // -0/0.1 = 0
    assert(float_equals(result->data[1], -1.0f / 0.8f)); // -1/0.8 = -1.25
    assert(float_equals(result->data[2], 0.0f));         // -0/0.1 = 0

    vector_free(prediction);
    vector_free(target);
    vector_free(result);
    TEST_PASSED;
}

void test_vector_cross_entropy_derivative_one_hot() {
    Vector *prediction = vector_create(4);
    Vector *target = vector_create(4);
    Vector *result = vector_create(4);

    // One-hot target: class 2
    target->data[0] = 0.0f;
    target->data[1] = 0.0f;
    target->data[2] = 1.0f;
    target->data[3] = 0.0f;

    prediction->data[0] = 0.2f;
    prediction->data[1] = 0.3f;
    prediction->data[2] = 0.4f;
    prediction->data[3] = 0.1f;

    vector_cross_entropy_derivative(result, prediction, target);

    // Only the target class should have non-zero derivative
    assert(float_equals(result->data[0], 0.0f));
    assert(float_equals(result->data[1], 0.0f));
    assert(float_equals(result->data[2], -1.0f / 0.4f));
    assert(float_equals(result->data[3], 0.0f));

    vector_free(prediction);
    vector_free(target);
    vector_free(result);
    TEST_PASSED;
}

void test_vector_cross_entropy_derivative_epsilon_protection() {
    Vector *prediction = vector_create(2);
    Vector *target = vector_create(2);
    Vector *result = vector_create(2);

    target->data[0] = 1.0f;
    target->data[1] = 0.0f;

    // Very small prediction to test epsilon
    prediction->data[0] = 0.0f;
    prediction->data[1] = 1.0f;

    vector_cross_entropy_derivative(result, prediction, target);

    // Should not crash or return inf/nan
    assert(isfinite(result->data[0]));
    assert(isfinite(result->data[1]));

    vector_free(prediction);
    vector_free(target);
    vector_free(result);
    TEST_PASSED;
}

void test_vector_cross_entropy_derivative_in_place() {
    Vector *prediction = vector_create(2);
    Vector *target = vector_create(2);

    target->data[0] = 1.0f;
    target->data[1] = 0.0f;

    prediction->data[0] = 0.5f;
    prediction->data[1] = 0.5f;

    // Use prediction as both input and output
    vector_cross_entropy_derivative(prediction, prediction, target);

    assert(float_equals(prediction->data[0], -2.0f)); // -1/0.5
    assert(float_equals(prediction->data[1], 0.0f));  // -0/0.5

    vector_free(prediction);
    vector_free(target);
    TEST_PASSED;
}

// =============================================================================
// Loss Function Pair Tests
// =============================================================================

void test_vector_mse_loss_pair() {
    Vector *prediction = vector_create(3);
    Vector *target = vector_create(3);
    Vector *derivative = vector_create(3);

    prediction->data[0] = 2.0f;
    prediction->data[1] = 3.0f;
    prediction->data[2] = 4.0f;

    target->data[0] = 1.0f;
    target->data[1] = 1.0f;
    target->data[2] = 1.0f;

    // Test that the loss pair struct works
    float loss = VECTOR_MSE_LOSS.loss(prediction, target);
    VECTOR_MSE_LOSS.loss_derivative(derivative, prediction, target);

    // MSE = ((2-1)^2 + (3-1)^2 + (4-1)^2) / 3 = (1 + 4 + 9) / 3 = 14/3
    assert(float_equals(loss, 14.0f / 3.0f));

    // Derivatives: [(2-1)/3, (3-1)/3, (4-1)/3] = [1/3, 2/3, 1]
    assert(float_equals(derivative->data[0], 1.0f / 3.0f));
    assert(float_equals(derivative->data[1], 2.0f / 3.0f));
    assert(float_equals(derivative->data[2], 1.0f));

    vector_free(prediction);
    vector_free(target);
    vector_free(derivative);
    TEST_PASSED;
}

void test_vector_cross_entropy_loss_pair() {
    Vector *prediction = vector_create(2);
    Vector *target = vector_create(2);
    Vector *derivative = vector_create(2);

    target->data[0] = 1.0f;
    target->data[1] = 0.0f;

    prediction->data[0] = 0.9f;
    prediction->data[1] = 0.1f;

    // Test that the loss pair struct works
    float loss = VECTOR_CROSS_ENTROPY_LOSS.loss(prediction, target);
    VECTOR_CROSS_ENTROPY_LOSS.loss_derivative(derivative, prediction, target);

    // CE = -log(0.9)
    assert(float_equals(loss, -logf(0.9f)));

    // Derivatives: [-1/0.9, 0]
    assert(float_equals(derivative->data[0], -1.0f / 0.9f));
    assert(float_equals(derivative->data[1], 0.0f));

    vector_free(prediction);
    vector_free(target);
    vector_free(derivative);
    TEST_PASSED;
}

// =============================================================================
// Edge Cases Tests
// =============================================================================

void test_loss_single_element_vectors() {
    Vector *prediction = vector_create(1);
    Vector *target = vector_create(1);
    Vector *derivative = vector_create(1);

    prediction->data[0] = 3.0f;
    target->data[0] = 1.0f;

    // Test MSE
    float mse_loss_val = vector_mse(prediction, target);
    assert(float_equals(mse_loss_val, 4.0f)); // (3-1)^2 = 4

    vector_mse_derivative(derivative, prediction, target);
    assert(float_equals(derivative->data[0], 2.0f)); // (3-1)/1 = 2

    // Test cross-entropy
    prediction->data[0] = 0.8f;
    target->data[0] = 1.0f;

    float ce_loss_val = vector_cross_entropy(prediction, target);
    assert(float_equals(ce_loss_val, -logf(0.8f)));

    vector_cross_entropy_derivative(derivative, prediction, target);
    assert(float_equals(derivative->data[0], -1.0f / 0.8f));

    vector_free(prediction);
    vector_free(target);
    vector_free(derivative);
    TEST_PASSED;
}

void test_mse_symmetry() {
    // MSE should be symmetric: loss(a,b) == loss(b,a)
    float loss1 = mse_loss(3.0f, 1.0f);
    float loss2 = mse_loss(1.0f, 3.0f);
    assert(float_equals(loss1, loss2));
    TEST_PASSED;
}

void test_loss_all_zeros() {
    Vector *prediction = vector_zeros(3);
    Vector *target = vector_zeros(3);
    Vector *derivative = vector_create(3);

    // Zero vectors should give zero loss and derivative
    float mse_loss_val = vector_mse(prediction, target);
    assert(float_equals(mse_loss_val, 0.0f));

    vector_mse_derivative(derivative, prediction, target);
    for (int i = 0; i < 3; i++) {
        assert(float_equals(derivative->data[i], 0.0f));
    }

    vector_free(prediction);
    vector_free(target);
    vector_free(derivative);
    TEST_PASSED;
}

// =============================================================================
// Test Runner
// =============================================================================

void run_loss_tests(void) {
    printf("\n=== Singular MSE Loss Tests ===\n");
    test_mse_loss_zero_error();
    test_mse_loss_basic();
    test_mse_loss_negative_error();
    test_mse_loss_fractional();

    printf("\n=== Vector MSE Loss Tests ===\n");
    test_vector_mse_zero_error();
    test_vector_mse_basic();
    test_vector_mse_mixed_errors();
    test_vector_mse_single_element();

    printf("\n=== Vector MSE Derivative Tests ===\n");
    test_vector_mse_derivative_zero_error();
    test_vector_mse_derivative_basic();
    test_vector_mse_derivative_negative_errors();
    test_vector_mse_derivative_in_place();

    printf("\n=== Vector Cross-Entropy Loss Tests ===\n");
    test_vector_cross_entropy_basic();
    test_vector_cross_entropy_perfect_prediction();
    test_vector_cross_entropy_multi_class();
    test_vector_cross_entropy_epsilon_protection();

    printf("\n=== Vector Cross-Entropy Derivative Tests ===\n");
    test_vector_cross_entropy_derivative_basic();
    test_vector_cross_entropy_derivative_one_hot();
    test_vector_cross_entropy_derivative_epsilon_protection();
    test_vector_cross_entropy_derivative_in_place();

    printf("\n=== Loss Function Pair Tests ===\n");
    test_vector_mse_loss_pair();
    test_vector_cross_entropy_loss_pair();

    printf("\n=== Loss Edge Cases Tests ===\n");
    test_loss_single_element_vectors();
    test_mse_symmetry();
    test_loss_all_zeros();

    printf("\n=== All Loss Tests Passed! ===\n");
}
