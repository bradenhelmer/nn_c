/*
 * test_optimizer.c - Comprehensive tests for optimizer implementations
 */

#include "../src/activations/activations.h"
#include "../src/nn/loss.h"
#include "../src/training/optimizer.h"
#include "test_runner.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// =============================================================================
// Helper Functions
// =============================================================================

// Create a simple 2-layer MLP for testing
static MLP *create_test_mlp(void) {
    // Create a simple network: 3 -> 4 -> 2
    MLP *mlp = mlp_create(2, 0.01f, VECTOR_MSE_LOSS, NULL);

    Layer *layer1 = layer_create(3, 4, VECTOR_TANH_ACTIVATION);
    Layer *layer2 = layer_create(4, 2, VECTOR_SIGMOID_ACTIVATION);

    mlp_add_layer(mlp, 0, layer1);
    mlp_add_layer(mlp, 1, layer2);

    return mlp;
}

// Set up gradients for testing (simulate a backward pass)
static void set_test_gradients(MLP *mlp) {
    for (int i = 0; i < mlp->num_layers; i++) {
        Layer *layer = mlp->layers[i];

        // Set some non-zero gradients
        for (int row = 0; row < layer->dW->rows; row++) {
            for (int col = 0; col < layer->dW->cols; col++) {
                matrix_set(layer->dW, row, col, 0.1f * (row + col + 1));
            }
        }

        for (int j = 0; j < layer->db->size; j++) {
            layer->db->data[j] = 0.1f * (j + 1);
        }
    }
}

// =============================================================================
// Creation and Destruction Tests
// =============================================================================

void test_optimizer_create_sgd() {
    Optimizer *opt = optimizer_create_sgd(0.01f);

    assert(opt != NULL);
    assert(opt->type == OPTIMIZER_SGD);
    assert(float_equals(opt->learning_rate, 0.01f));
    assert(opt->v_weights == NULL);
    assert(opt->v_biases == NULL);

    optimizer_free(opt);
    TEST_PASSED;
}

void test_optimizer_create_momentum() {
    Optimizer *opt = optimizer_create_momentum(0.01f, 0.9f);

    assert(opt != NULL);
    assert(opt->type == OPTIMIZER_MOMENTUM);
    assert(float_equals(opt->learning_rate, 0.01f));
    assert(float_equals(opt->beta, 0.9f));
    // v_weights and v_biases should be NULL until init
    assert(opt->v_weights == NULL);
    assert(opt->v_biases == NULL);

    optimizer_free(opt);
    TEST_PASSED;
}

void test_optimizer_create_adam() {
    Optimizer *opt = optimizer_create_adam(0.001f, 0.9f, 0.999f, 1e-8f);

    assert(opt != NULL);
    assert(opt->type == OPTIMIZER_MOMENTUM);
    assert(float_equals(opt->learning_rate, 0.001f));
    assert(float_equals(opt->beta, 0.9f));
    assert(float_equals(opt->beta1, 0.9f));
    assert(float_equals(opt->beta2, 0.999f));
    assert(float_equals(opt->epsilon, 1e-8f));
    assert(opt->timestep == 0);

    optimizer_free(opt);
    TEST_PASSED;
}

// =============================================================================
// Initialization Tests
// =============================================================================

void test_optimizer_init_sgd() {
    MLP *mlp = create_test_mlp();
    Optimizer *opt = optimizer_create_sgd(0.01f);

    optimizer_init(opt, mlp);

    assert(opt->num_layers == mlp->num_layers);
    // SGD shouldn't allocate velocity arrays
    assert(opt->v_weights == NULL);
    assert(opt->v_biases == NULL);

    optimizer_free(opt);
    mlp_free(mlp);
    TEST_PASSED;
}

void test_optimizer_init_momentum() {
    MLP *mlp = create_test_mlp();
    Optimizer *opt = optimizer_create_momentum(0.01f, 0.9f);

    optimizer_init(opt, mlp);

    assert(opt->num_layers == mlp->num_layers);
    assert(opt->v_weights != NULL);
    assert(opt->v_biases != NULL);

    // Check that velocity matrices are allocated with correct dimensions
    for (int i = 0; i < mlp->num_layers; i++) {
        Layer *layer = mlp->layers[i];
        assert(opt->v_weights[i] != NULL);
        assert(opt->v_weights[i]->rows == layer->weights->rows);
        assert(opt->v_weights[i]->cols == layer->weights->cols);

        assert(opt->v_biases[i] != NULL);
        assert(opt->v_biases[i]->size == layer->biases->size);

        // Check that velocities are initialized to zero
        for (int row = 0; row < opt->v_weights[i]->rows; row++) {
            for (int col = 0; col < opt->v_weights[i]->cols; col++) {
                assert(float_equals(matrix_get(opt->v_weights[i], row, col), 0.0f));
            }
        }

        for (int j = 0; j < opt->v_biases[i]->size; j++) {
            assert(float_equals(opt->v_biases[i]->data[j], 0.0f));
        }
    }

    optimizer_free(opt);
    mlp_free(mlp);
    TEST_PASSED;
}

// =============================================================================
// Learning Rate Tests
// =============================================================================

void test_optimizer_get_set_lr() {
    Optimizer *opt = optimizer_create_sgd(0.01f);

    assert(float_equals(optimizer_get_lr(opt), 0.01f));

    optimizer_set_lr(opt, 0.001f);
    assert(float_equals(optimizer_get_lr(opt), 0.001f));

    optimizer_set_lr(opt, 1.0f);
    assert(float_equals(optimizer_get_lr(opt), 1.0f));

    optimizer_free(opt);
    TEST_PASSED;
}

// =============================================================================
// SGD Optimizer Step Tests
// =============================================================================

void test_optimizer_step_sgd_basic() {
    MLP *mlp = create_test_mlp();
    Optimizer *opt = optimizer_create_sgd(0.1f);
    optimizer_init(opt, mlp);

    // Store original weights
    float original_weight = matrix_get(mlp->layers[0]->weights, 0, 0);
    float original_bias = mlp->layers[0]->biases->data[0];

    // Set gradients
    set_test_gradients(mlp);
    float gradient_w = matrix_get(mlp->layers[0]->dW, 0, 0);
    float gradient_b = mlp->layers[0]->db->data[0];

    // Step should perform: W = W - lr * dW
    optimizer_step(opt, mlp);

    float expected_weight = original_weight - 0.1f * gradient_w;
    float expected_bias = original_bias - 0.1f * gradient_b;

    assert(float_equals(matrix_get(mlp->layers[0]->weights, 0, 0), expected_weight));
    assert(float_equals(mlp->layers[0]->biases->data[0], expected_bias));

    optimizer_free(opt);
    mlp_free(mlp);
    TEST_PASSED;
}

void test_optimizer_step_sgd_zero_lr() {
    MLP *mlp = create_test_mlp();
    Optimizer *opt = optimizer_create_sgd(0.0f);
    optimizer_init(opt, mlp);

    float original_weight = matrix_get(mlp->layers[0]->weights, 0, 0);
    set_test_gradients(mlp);

    // With zero learning rate, weights shouldn't change
    optimizer_step(opt, mlp);

    assert(float_equals(matrix_get(mlp->layers[0]->weights, 0, 0), original_weight));

    optimizer_free(opt);
    mlp_free(mlp);
    TEST_PASSED;
}

// =============================================================================
// Momentum Optimizer Step Tests
// =============================================================================

void test_optimizer_step_momentum_first_step() {
    MLP *mlp = create_test_mlp();
    Optimizer *opt = optimizer_create_momentum(0.1f, 0.9f);
    optimizer_init(opt, mlp);

    float original_weight = matrix_get(mlp->layers[0]->weights, 0, 0);
    float original_bias = mlp->layers[0]->biases->data[0];

    set_test_gradients(mlp);
    float gradient_w = matrix_get(mlp->layers[0]->dW, 0, 0);
    float gradient_b = mlp->layers[0]->db->data[0];

    // First step: v = 0.9 * 0 + dW = dW, W = W - lr * v
    optimizer_step(opt, mlp);

    // Check velocity was updated correctly
    assert(float_equals(matrix_get(opt->v_weights[0], 0, 0), gradient_w));
    assert(float_equals(opt->v_biases[0]->data[0], gradient_b));

    // Check weights were updated correctly
    float expected_weight = original_weight - 0.1f * gradient_w;
    float expected_bias = original_bias - 0.1f * gradient_b;

    assert(float_equals(matrix_get(mlp->layers[0]->weights, 0, 0), expected_weight));
    assert(float_equals(mlp->layers[0]->biases->data[0], expected_bias));

    optimizer_free(opt);
    mlp_free(mlp);
    TEST_PASSED;
}

void test_optimizer_step_momentum_second_step() {
    MLP *mlp = create_test_mlp();
    Optimizer *opt = optimizer_create_momentum(0.1f, 0.9f);
    optimizer_init(opt, mlp);

    set_test_gradients(mlp);
    float gradient_w = matrix_get(mlp->layers[0]->dW, 0, 0);

    // First step
    optimizer_step(opt, mlp);
    float weight_after_first = matrix_get(mlp->layers[0]->weights, 0, 0);
    float velocity_after_first = matrix_get(opt->v_weights[0], 0, 0);

    // Second step with same gradients
    set_test_gradients(mlp);
    optimizer_step(opt, mlp);

    // v = 0.9 * velocity_after_first + gradient_w
    float expected_velocity = 0.9f * velocity_after_first + gradient_w;
    assert(float_equals(matrix_get(opt->v_weights[0], 0, 0), expected_velocity));

    // W = weight_after_first - 0.1 * expected_velocity
    float expected_weight = weight_after_first - 0.1f * expected_velocity;
    assert(float_equals(matrix_get(mlp->layers[0]->weights, 0, 0), expected_weight));

    optimizer_free(opt);
    mlp_free(mlp);
    TEST_PASSED;
}

void test_optimizer_step_momentum_accumulation() {
    MLP *mlp = create_test_mlp();
    Optimizer *opt = optimizer_create_momentum(0.01f, 0.9f);
    optimizer_init(opt, mlp);

    float original_weight = matrix_get(mlp->layers[0]->weights, 0, 0);

    // Run multiple steps - velocity should accumulate
    for (int step = 0; step < 5; step++) {
        set_test_gradients(mlp);
        optimizer_step(opt, mlp);
    }

    // Weight should have moved more than a single SGD step would
    float final_weight = matrix_get(mlp->layers[0]->weights, 0, 0);
    float gradient_w = 0.1f; // From set_test_gradients for (0,0)
    float single_sgd_step = original_weight - 0.01f * gradient_w;

    // Momentum should cause larger change
    assert(fabs(final_weight - original_weight) > fabs(single_sgd_step - original_weight));

    optimizer_free(opt);
    mlp_free(mlp);
    TEST_PASSED;
}

void test_optimizer_step_momentum_beta_zero() {
    MLP *mlp = create_test_mlp();
    Optimizer *opt = optimizer_create_momentum(0.1f, 0.0f); // beta = 0 => SGD
    optimizer_init(opt, mlp);

    float original_weight = matrix_get(mlp->layers[0]->weights, 0, 0);
    set_test_gradients(mlp);
    float gradient_w = matrix_get(mlp->layers[0]->dW, 0, 0);

    optimizer_step(opt, mlp);

    // With beta=0, should behave like SGD
    float expected_weight = original_weight - 0.1f * gradient_w;
    assert(float_equals(matrix_get(mlp->layers[0]->weights, 0, 0), expected_weight));

    optimizer_free(opt);
    mlp_free(mlp);
    TEST_PASSED;
}

// =============================================================================
// Multi-layer Tests
// =============================================================================

void test_optimizer_step_all_layers() {
    MLP *mlp = create_test_mlp();
    Optimizer *opt = optimizer_create_momentum(0.1f, 0.9f);
    optimizer_init(opt, mlp);

    // Store original weights for all layers
    float original_weights[2];
    for (int i = 0; i < mlp->num_layers; i++) {
        original_weights[i] = matrix_get(mlp->layers[i]->weights, 0, 0);
    }

    set_test_gradients(mlp);
    optimizer_step(opt, mlp);

    // Check that all layers were updated
    for (int i = 0; i < mlp->num_layers; i++) {
        float new_weight = matrix_get(mlp->layers[i]->weights, 0, 0);
        assert(!float_equals(new_weight, original_weights[i]));
    }

    optimizer_free(opt);
    mlp_free(mlp);
    TEST_PASSED;
}

// =============================================================================
// Edge Cases Tests
// =============================================================================

void test_optimizer_step_zero_gradients() {
    MLP *mlp = create_test_mlp();
    Optimizer *opt = optimizer_create_sgd(0.1f);
    optimizer_init(opt, mlp);

    float original_weight = matrix_get(mlp->layers[0]->weights, 0, 0);

    // Don't set gradients - they should be zero
    mlp_zero_gradients(mlp);
    optimizer_step(opt, mlp);

    // Weights shouldn't change with zero gradients
    assert(float_equals(matrix_get(mlp->layers[0]->weights, 0, 0), original_weight));

    optimizer_free(opt);
    mlp_free(mlp);
    TEST_PASSED;
}

void test_optimizer_step_negative_gradients() {
    MLP *mlp = create_test_mlp();
    Optimizer *opt = optimizer_create_sgd(0.1f);
    optimizer_init(opt, mlp);

    float original_weight = matrix_get(mlp->layers[0]->weights, 0, 0);

    // Set negative gradients
    matrix_set(mlp->layers[0]->dW, 0, 0, -1.0f);

    optimizer_step(opt, mlp);

    // W = W - lr * (-1.0) = W + lr
    float expected = original_weight + 0.1f;
    assert(float_equals(matrix_get(mlp->layers[0]->weights, 0, 0), expected));

    optimizer_free(opt);
    mlp_free(mlp);
    TEST_PASSED;
}

// =============================================================================
// Test Runner
// =============================================================================

void run_optimizer_tests(void) {
    printf("\n=== Optimizer Creation and Destruction Tests ===\n");
    test_optimizer_create_sgd();
    test_optimizer_create_momentum();
    test_optimizer_create_adam();

    printf("\n=== Optimizer Initialization Tests ===\n");
    test_optimizer_init_sgd();
    test_optimizer_init_momentum();

    printf("\n=== Optimizer Learning Rate Tests ===\n");
    test_optimizer_get_set_lr();

    printf("\n=== SGD Optimizer Step Tests ===\n");
    test_optimizer_step_sgd_basic();
    test_optimizer_step_sgd_zero_lr();

    printf("\n=== Momentum Optimizer Step Tests ===\n");
    test_optimizer_step_momentum_first_step();
    test_optimizer_step_momentum_second_step();
    test_optimizer_step_momentum_accumulation();
    test_optimizer_step_momentum_beta_zero();

    printf("\n=== Multi-layer Optimizer Tests ===\n");
    test_optimizer_step_all_layers();

    printf("\n=== Optimizer Edge Cases Tests ===\n");
    test_optimizer_step_zero_gradients();
    test_optimizer_step_negative_gradients();

    printf("\n=== All Optimizer Tests Passed! ===\n");
}
