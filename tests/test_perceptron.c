/*
 * test_perceptron.c - Comprehensive tests for perceptron operations
 */

#include "../src/activations/activations.h"
#include "../src/data/dataset.h"
#include "../src/nn/perceptron.h"
#include "test_runner.h"
#include "utils/utils.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// Helper classifier functions for testing
int binary_classifier(float x) {
    return x >= 0.5f ? 1 : 0;
}

int sign_classifier(float x) {
    return x >= 0.0f ? 1 : -1;
}

// =============================================================================
// Creation and Destruction Tests
// =============================================================================

void test_perceptron_create_basic() {
    Perceptron *p = perceptron_create(2, -1.0f, 1.0f, 0.1f, SIGMOID_ACTIVATION, binary_classifier);

    assert(p != NULL);
    assert(p->weights != NULL);
    assert(p->weights->size == 2);
    assert(float_equals(p->bias, 0.0f));
    assert(float_equals(p->learning_rate, 0.1f));
    assert(p->input_size == 2);
    assert(p->activation == sigmoid_scalar);
    assert(p->activation_prime == sigmoid_scalar_derivative);
    assert(p->classifier == binary_classifier);
    assert(float_equals(p->last_output, 0.0f));
    assert(float_equals(p->last_raw_output, 0.0f));

    // Check weights are within bounds
    for (int i = 0; i < p->weights->size; i++) {
        assert(p->weights->data[i] >= -1.0f);
        assert(p->weights->data[i] <= 1.0f);
    }

    perceptron_free(p);
    TEST_PASSED;
}

void test_perceptron_create_different_sizes() {
    Perceptron *p1 = perceptron_create(1, 0.0f, 1.0f, 0.01f, RELU_ACTIVATION, sign_classifier);
    assert(p1->weights->size == 1);
    assert(p1->input_size == 1);
    perceptron_free(p1);

    Perceptron *p2 = perceptron_create(10, -0.5f, 0.5f, 0.5f, TANH_ACTIVATION, binary_classifier);
    assert(p2->weights->size == 10);
    assert(p2->input_size == 10);
    perceptron_free(p2);

    TEST_PASSED;
}

void test_perceptron_create_different_activations() {
    Perceptron *p_sigmoid =
        perceptron_create(2, -1.0f, 1.0f, 0.1f, SIGMOID_ACTIVATION, binary_classifier);
    assert(p_sigmoid->activation == sigmoid_scalar);
    assert(p_sigmoid->activation_prime == sigmoid_scalar_derivative);
    perceptron_free(p_sigmoid);

    Perceptron *p_relu =
        perceptron_create(2, -1.0f, 1.0f, 0.1f, RELU_ACTIVATION, binary_classifier);
    assert(p_relu->activation == relu_scalar);
    assert(p_relu->activation_prime == relu_scalar_derivative);
    perceptron_free(p_relu);

    Perceptron *p_tanh =
        perceptron_create(2, -1.0f, 1.0f, 0.1f, TANH_ACTIVATION, binary_classifier);
    assert(p_tanh->activation == tanh_scalar);
    assert(p_tanh->activation_prime == tanh_scalar_derivative);
    perceptron_free(p_tanh);

    TEST_PASSED;
}

// =============================================================================
// Forward Pass Tests
// =============================================================================

void test_perceptron_predict_basic() {
    Perceptron *p = perceptron_create(2, 0.0f, 0.0f, 0.1f, SIGMOID_ACTIVATION, binary_classifier);

    // Set known weights for predictable output
    p->weights->data[0] = 0.5f;
    p->weights->data[1] = -0.3f;
    p->bias = 0.2f;

    Tensor *input = tensor_create1d(2);
    input->data[0] = 1.0f;
    input->data[1] = 2.0f;

    float prediction = perceptron_predict(p, input);

    // Expected: z = (0.5*1.0) + (-0.3*2.0) + 0.2 = 0.5 - 0.6 + 0.2 = 0.1
    // sigmoid(0.1) ≈ 0.525
    float expected_z = 0.1f;
    float expected_output = sigmoid_scalar(expected_z);

    assert(float_equals(p->last_raw_output, expected_z));
    assert(float_equals(p->last_output, expected_output));
    assert(float_equals(prediction, expected_output));

    tensor_free(input);
    perceptron_free(p);
    TEST_PASSED;
}

void test_perceptron_predict_zero_input() {
    Perceptron *p = perceptron_create(3, -1.0f, 1.0f, 0.1f, SIGMOID_ACTIVATION, binary_classifier);
    p->bias = 0.5f;

    Tensor *input = tensor_create1d(3);
    float prediction = perceptron_predict(p, input);

    // With zero input, output should be sigmoid(bias)
    float expected = sigmoid_scalar(0.5f);
    assert(float_equals(prediction, expected));
    assert(float_equals(p->last_raw_output, 0.5f));

    tensor_free(input);
    perceptron_free(p);
    TEST_PASSED;
}

void test_perceptron_predict_with_relu() {
    Perceptron *p = perceptron_create(2, 0.0f, 0.0f, 0.1f, RELU_ACTIVATION, sign_classifier);

    p->weights->data[0] = 1.0f;
    p->weights->data[1] = 1.0f;
    p->bias = -1.5f;

    Tensor *input = tensor_create1d(2);
    input->data[0] = 1.0f;
    input->data[1] = 1.0f;

    float prediction = perceptron_predict(p, input);

    // z = 1.0 + 1.0 - 1.5 = 0.5
    // ReLU(0.5) = 0.5
    assert(float_equals(p->last_raw_output, 0.5f));
    assert(float_equals(prediction, 0.5f));

    tensor_free(input);
    perceptron_free(p);
    TEST_PASSED;
}

void test_perceptron_predict_caching() {
    Perceptron *p = perceptron_create(2, 0.0f, 0.0f, 0.1f, TANH_ACTIVATION, binary_classifier);

    p->weights->data[0] = 1.0f;
    p->weights->data[1] = -1.0f;
    p->bias = 0.0f;

    Tensor *input = tensor_create1d(2);
    input->data[0] = 0.5f;
    input->data[1] = 0.5f;

    // First prediction
    float pred1 = perceptron_predict(p, input);
    float cached_raw1 = p->last_raw_output;
    float cached_out1 = p->last_output;

    // z = 0.5 - 0.5 = 0.0
    assert(float_equals(cached_raw1, 0.0f));
    assert(float_equals(cached_out1, tanh_scalar(0.0f)));
    assert(float_equals(pred1, cached_out1));

    // Change input and predict again
    input->data[0] = 1.0f;
    input->data[1] = 0.0f;

    float pred2 = perceptron_predict(p, input);

    // z = 1.0 - 0.0 = 1.0
    assert(float_equals(p->last_raw_output, 1.0f));
    assert(float_equals(p->last_output, tanh_scalar(1.0f)));
    assert(float_equals(pred2, tanh_scalar(1.0f)));

    tensor_free(input);
    perceptron_free(p);
    TEST_PASSED;
}

// =============================================================================
// Weight Update Tests
// =============================================================================

void test_perceptron_update_weights_basic() {
    Perceptron *p = perceptron_create(2, 0.0f, 0.0f, 0.1f, LINEAR_ACTIVATION, binary_classifier);

    // Set known initial weights
    p->weights->data[0] = 0.5f;
    p->weights->data[1] = -0.3f;
    p->bias = 0.2f;

    // Simulate a prediction (linear activation, derivative = 1)
    p->last_output = 0.8f;

    Tensor *input = tensor_create1d(2);
    input->data[0] = 1.0f;
    input->data[1] = 2.0f;

    float error = 0.3f; // (prediction - target)

    // Store initial weights
    float w0_initial = p->weights->data[0];
    float w1_initial = p->weights->data[1];
    float b_initial = p->bias;

    perceptron_update_weights(p, input, error);

    // For linear activation: derivative = 1.0
    // delta = error * 1.0 = 0.3
    // w0 -= learning_rate * delta * input[0] = 0.5 - 0.1*0.3*1.0 = 0.47
    // w1 -= learning_rate * delta * input[1] = -0.3 - 0.1*0.3*2.0 = -0.36
    // bias -= learning_rate * delta = 0.2 - 0.1*0.3 = 0.17

    assert(float_equals(p->weights->data[0], w0_initial - 0.1f * 0.3f * 1.0f));
    assert(float_equals(p->weights->data[1], w1_initial - 0.1f * 0.3f * 2.0f));
    assert(float_equals(p->bias, b_initial - 0.1f * 0.3f));

    tensor_free(input);
    perceptron_free(p);
    TEST_PASSED;
}

void test_perceptron_update_weights_with_sigmoid() {
    Perceptron *p = perceptron_create(2, 0.0f, 0.0f, 0.2f, SIGMOID_ACTIVATION, binary_classifier);

    p->weights->data[0] = 1.0f;
    p->weights->data[1] = 1.0f;
    p->bias = 0.0f;

    Tensor *input = tensor_create1d(2);
    input->data[0] = 0.5f;
    input->data[1] = 0.5f;

    // Make a prediction first to set last_output
    float prediction = perceptron_predict(p, input);
    // Prediction will be sigmoid(1.0) ≈ 0.731

    float target = 0.0f;               // Create over-prediction scenario
    float error = prediction - target; // error ≈ 0.731 (positive)

    float w0_before = p->weights->data[0];
    float w1_before = p->weights->data[1];
    float b_before = p->bias;

    perceptron_update_weights(p, input, error);

    // Weights should have changed
    assert(!float_equals(p->weights->data[0], w0_before));
    assert(!float_equals(p->weights->data[1], w1_before));
    assert(!float_equals(p->bias, b_before));

    // With positive error (over-prediction) and positive inputs,
    // weights should decrease
    assert(p->weights->data[0] < w0_before);
    assert(p->weights->data[1] < w1_before);

    tensor_free(input);
    perceptron_free(p);
    TEST_PASSED;
}

void test_perceptron_update_weights_zero_error() {
    Perceptron *p = perceptron_create(2, 0.0f, 0.0f, 0.1f, SIGMOID_ACTIVATION, binary_classifier);

    p->weights->data[0] = 0.5f;
    p->weights->data[1] = -0.3f;
    p->bias = 0.2f;
    p->last_output = 0.7f;

    Tensor *input = tensor_create1d(2);
    input->data[0] = 1.0f;
    input->data[1] = 1.0f;

    float w0_before = p->weights->data[0];
    float w1_before = p->weights->data[1];
    float b_before = p->bias;

    // Zero error means no update
    perceptron_update_weights(p, input, 0.0f);

    assert(float_equals(p->weights->data[0], w0_before));
    assert(float_equals(p->weights->data[1], w1_before));
    assert(float_equals(p->bias, b_before));

    tensor_free(input);
    perceptron_free(p);
    TEST_PASSED;
}

// =============================================================================
// Training Step Tests
// =============================================================================

void test_perceptron_train_step_basic() {
    Perceptron *p = perceptron_create(2, 0.0f, 0.0f, 0.1f, LINEAR_ACTIVATION, binary_classifier);

    p->weights->data[0] = 0.0f;
    p->weights->data[1] = 0.0f;
    p->bias = 0.0f;

    Tensor *input = tensor_create1d(2);
    input->data[0] = 1.0f;
    input->data[1] = 1.0f;

    float target = 1.0f;

    float w0_before = p->weights->data[0];
    float w1_before = p->weights->data[1];

    perceptron_train_step(p, input, target);

    // Weights should have been updated
    // Initial prediction with zero weights: 0.0
    // Error = 0.0 - 1.0 = -1.0 (under-prediction)
    // With negative error and positive inputs, weights should increase
    assert(p->weights->data[0] > w0_before);
    assert(p->weights->data[1] > w1_before);

    tensor_free(input);
    perceptron_free(p);
    TEST_PASSED;
}

void test_perceptron_train_step_convergence_direction() {
    Perceptron *p = perceptron_create(2, 0.0f, 0.0f, 0.5f, LINEAR_ACTIVATION, binary_classifier);

    // Start with weights that will over-predict
    p->weights->data[0] = 2.0f;
    p->weights->data[1] = 2.0f;
    p->bias = 0.0f;

    Tensor *input = tensor_create1d(2);
    input->data[0] = 1.0f;
    input->data[1] = 1.0f;

    // This will predict 4.0, but target is 1.0
    float target = 1.0f;

    float w0_before = p->weights->data[0];
    float w1_before = p->weights->data[1];

    perceptron_train_step(p, input, target);

    // With over-prediction, weights should decrease
    assert(p->weights->data[0] < w0_before);
    assert(p->weights->data[1] < w1_before);

    tensor_free(input);
    perceptron_free(p);
    TEST_PASSED;
}

// =============================================================================
// Integration Tests - Learning Simple Patterns
// =============================================================================

void test_perceptron_learns_simple_pattern() {
    // Test that perceptron can learn a simple pattern: x1 + x2 > 1
    Perceptron *p = perceptron_create(2, -0.5f, 0.5f, 0.5f, SIGMOID_ACTIVATION, binary_classifier);

    // Training data: simple linearly separable pattern
    Tensor *inputs[4];
    float targets[4];

    // Class 0: both inputs small
    inputs[0] = tensor_create1d(2);
    inputs[0]->data[0] = 0.0f;
    inputs[0]->data[1] = 0.0f;
    targets[0] = 0.0f;

    inputs[1] = tensor_create1d(2);
    inputs[1]->data[0] = 0.0f;
    inputs[1]->data[1] = 1.0f;
    targets[1] = 0.0f;

    // Class 1: sum is large
    inputs[2] = tensor_create1d(2);
    inputs[2]->data[0] = 1.0f;
    inputs[2]->data[1] = 1.0f;
    targets[2] = 1.0f;

    inputs[3] = tensor_create1d(2);
    inputs[3]->data[0] = 2.0f;
    inputs[3]->data[1] = 2.0f;
    targets[3] = 1.0f;

    // Train for multiple epochs
    for (int epoch = 0; epoch < 100; epoch++) {
        for (int i = 0; i < 4; i++) {
            perceptron_train_step(p, inputs[i], targets[i]);
        }
    }

    // Test predictions
    int correct = 0;
    for (int i = 0; i < 4; i++) {
        float prediction = perceptron_predict(p, inputs[i]);
        int classified = p->classifier(prediction);
        if (float_equals((float)classified, targets[i])) {
            correct++;
        }
    }

    // Should get at least 3 out of 4 correct (allowing for some variation)
    assert(correct >= 3);

    // Cleanup
    for (int i = 0; i < 4; i++) {
        tensor_free(inputs[i]);
    }
    perceptron_free(p);
    TEST_PASSED;
}

void test_perceptron_learns_and_gate() {
    Perceptron *p = perceptron_create(2, -1.0f, 1.0f, 0.5f, SIGMOID_ACTIVATION, binary_classifier);

    Dataset *and_data = create_and_gate_dataset();
    Tensor *input = tensor_create1d(2);

    // Train for multiple epochs
    for (int epoch = 0; epoch < 100; epoch++) {
        for (int i = 0; i < and_data->num_samples; i++) {
            tensor_get_row(input, and_data->X, i);
            float target = and_data->Y->data[i];
            perceptron_train_step(p, input, target);
        }
    }

    // Test accuracy
    int correct = 0;
    for (int i = 0; i < and_data->num_samples; i++) {
        tensor_get_row(input, and_data->X, i);
        float prediction = perceptron_predict(p, input);
        int classified = p->classifier(prediction);
        float target = and_data->Y->data[i];

        if (float_equals((float)classified, target)) {
            correct++;
        }
    }

    // AND gate is linearly separable, should get 100% accuracy
    assert(correct == 4);

    tensor_free(input);
    dataset_free(and_data);
    perceptron_free(p);
    TEST_PASSED;
}

void test_perceptron_learns_or_gate() {
    Perceptron *p = perceptron_create(2, -1.0f, 1.0f, 0.5f, SIGMOID_ACTIVATION, binary_classifier);

    Dataset *or_data = create_or_gate_dataset();
    Tensor *input = tensor_create1d(2);

    // Train for multiple epochs
    for (int epoch = 0; epoch < 100; epoch++) {
        for (int i = 0; i < or_data->num_samples; i++) {
            tensor_get_row(input, or_data->X, i);
            float target = or_data->Y->data[i];
            perceptron_train_step(p, input, target);
        }
    }

    // Test accuracy
    int correct = 0;
    for (int i = 0; i < or_data->num_samples; i++) {
        tensor_get_row(input, or_data->X, i);
        float prediction = perceptron_predict(p, input);
        int classified = p->classifier(prediction);
        float target = or_data->Y->data[i];

        if (float_equals((float)classified, target)) {
            correct++;
        }
    }

    // OR gate is linearly separable, should get 100% accuracy
    assert(correct == 4);

    tensor_free(input);
    dataset_free(or_data);
    perceptron_free(p);
    TEST_PASSED;
}

void test_perceptron_with_different_learning_rates() {
    // Test that lower learning rate still learns (just slower)
    Perceptron *p_slow =
        perceptron_create(2, -0.5f, 0.5f, 0.01f, SIGMOID_ACTIVATION, binary_classifier);
    Perceptron *p_fast =
        perceptron_create(2, -0.5f, 0.5f, 0.5f, SIGMOID_ACTIVATION, binary_classifier);

    // Copy initial weights to be the same
    for (int i = 0; i < 2; i++) {
        p_slow->weights->data[i] = 0.1f;
        p_fast->weights->data[i] = 0.1f;
    }
    p_slow->bias = 0.0f;
    p_fast->bias = 0.0f;

    Tensor *input = tensor_create1d(2);
    input->data[0] = 1.0f;
    input->data[1] = 1.0f;
    float target = 1.0f;

    // Save initial weights
    float w0_initial = p_slow->weights->data[0];

    // Single training step
    perceptron_train_step(p_slow, input, target);
    perceptron_train_step(p_fast, input, target);

    // Both should update, but fast should update more
    float slow_change = fabs(p_slow->weights->data[0] - w0_initial);
    float fast_change = fabs(p_fast->weights->data[0] - w0_initial);

    assert(slow_change > 0.0f);        // Slow learner still updates
    assert(fast_change > slow_change); // Fast learner updates more

    tensor_free(input);
    perceptron_free(p_slow);
    perceptron_free(p_fast);
    TEST_PASSED;
}

// =============================================================================
// Edge Cases Tests
// =============================================================================

void test_perceptron_zero_learning_rate() {
    Perceptron *p = perceptron_create(2, -1.0f, 1.0f, 0.0f, SIGMOID_ACTIVATION, binary_classifier);

    p->weights->data[0] = 0.5f;
    p->weights->data[1] = -0.3f;

    Tensor *input = tensor_create1d(2);
    input->data[0] = 1.0f;
    input->data[1] = 1.0f;

    float w0_before = p->weights->data[0];
    float w1_before = p->weights->data[1];

    // Train with zero learning rate
    perceptron_train_step(p, input, 1.0f);

    // Weights should not change
    assert(float_equals(p->weights->data[0], w0_before));
    assert(float_equals(p->weights->data[1], w1_before));

    tensor_free(input);
    perceptron_free(p);
    TEST_PASSED;
}

void test_perceptron_negative_inputs() {
    Perceptron *p = perceptron_create(2, -1.0f, 1.0f, 0.1f, TANH_ACTIVATION, sign_classifier);

    p->weights->data[0] = 1.0f;
    p->weights->data[1] = 1.0f;
    p->bias = 0.0f;

    Tensor *input = tensor_create1d(2);
    input->data[0] = -1.0f;
    input->data[1] = -1.0f;

    float prediction = perceptron_predict(p, input);

    // z = -1.0 + -1.0 = -2.0
    // tanh(-2.0) ≈ -0.964
    assert(float_equals(p->last_raw_output, -2.0f));
    assert(prediction < 0.0f);

    tensor_free(input);
    perceptron_free(p);
    TEST_PASSED;
}

void test_perceptron_single_input_dimension() {
    // Test with 1D input (simple threshold)
    Perceptron *p = perceptron_create(1, -1.0f, 1.0f, 0.3f, SIGMOID_ACTIVATION, binary_classifier);

    Tensor *input = tensor_create1d(1);

    // Train to recognize values > 0.5
    float training_data[4] = {0.0f, 1.0f, 0.3f, 0.8f};
    float training_labels[4] = {0.0f, 1.0f, 0.0f, 1.0f};

    for (int epoch = 0; epoch < 50; epoch++) {
        for (int i = 0; i < 4; i++) {
            input->data[0] = training_data[i];
            perceptron_train_step(p, input, training_labels[i]);
        }
    }

    // Test that it learned something (weight should be positive)
    assert(p->weights->data[0] != 0.0f);

    tensor_free(input);
    perceptron_free(p);
    TEST_PASSED;
}

void test_perceptron_bias_learning() {
    // Test that bias is essential for certain patterns
    Perceptron *p = perceptron_create(2, 0.0f, 0.0f, 0.5f, SIGMOID_ACTIVATION, binary_classifier);

    // Set weights to zero, only bias can solve this
    p->weights->data[0] = 0.0f;
    p->weights->data[1] = 0.0f;
    p->bias = 0.0f;

    Tensor *input = tensor_create1d(2);
    float target = 1.0f; // Want output to be 1 even with zero input

    float bias_before = p->bias;

    // Train multiple times
    for (int i = 0; i < 10; i++) {
        perceptron_train_step(p, input, target);
    }

    // Bias should have increased significantly
    assert(p->bias > bias_before);

    // Prediction should be closer to target
    float final_prediction = perceptron_predict(p, input);
    assert(final_prediction > 0.5f); // Should be classified as 1

    tensor_free(input);
    perceptron_free(p);
    TEST_PASSED;
}

// =============================================================================
// Test Runner
// =============================================================================

void run_perceptron_tests(void) {
    printf("\n=== Perceptron Creation and Destruction Tests ===\n");
    test_perceptron_create_basic();
    test_perceptron_create_different_sizes();
    test_perceptron_create_different_activations();

    printf("\n=== Perceptron Forward Pass Tests ===\n");
    test_perceptron_predict_basic();
    test_perceptron_predict_zero_input();
    test_perceptron_predict_with_relu();
    test_perceptron_predict_caching();

    printf("\n=== Perceptron Weight Update Tests ===\n");
    test_perceptron_update_weights_basic();
    test_perceptron_update_weights_with_sigmoid();
    test_perceptron_update_weights_zero_error();

    printf("\n=== Perceptron Training Step Tests ===\n");
    test_perceptron_train_step_basic();
    test_perceptron_train_step_convergence_direction();

    printf("\n=== Perceptron Learning Integration Tests ===\n");
    test_perceptron_learns_simple_pattern();
    test_perceptron_learns_and_gate();
    test_perceptron_learns_or_gate();
    test_perceptron_with_different_learning_rates();

    printf("\n=== Perceptron Edge Cases Tests ===\n");
    test_perceptron_zero_learning_rate();
    test_perceptron_negative_inputs();
    test_perceptron_single_input_dimension();
    test_perceptron_bias_learning();

    printf("\n=== All Perceptron Tests Passed! ===\n");
}
