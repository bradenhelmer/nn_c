/*
 * test_loss.c - Comprehensive tests for loss functions
 */

#include "../src/nn/loss.h"
#include "test_runner.h"
#include "utils/utils.h"
#include <assert.h>
#include <stdio.h>

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
// Test Runner
// =============================================================================

void run_loss_tests(void) {
    printf("\n=== Singular MSE Loss Tests ===\n");
    test_mse_loss_zero_error();
    test_mse_loss_basic();
    test_mse_loss_negative_error();
    test_mse_loss_fractional();

    printf("\n=== All Loss Tests Passed! ===\n");
}
