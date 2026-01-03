/*
 * test_optimizer.c - Comprehensive tests for optimizer implementations
 */

#include "../src/activations/activations.h"
#include "../src/nn/loss.h"
#include "../src/training/optimizer.h"
#include "test_runner.h"
#include "utils/utils.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

// =============================================================================
// Helper Functions
// =============================================================================

// Create a simple 4-layer NeuralNet for testing (2 linear + 2 activation)
static NeuralNet *create_test_nn(void) {
    // Create a simple network: 3 -> Linear(4) -> Tanh -> Linear(2) -> Sigmoid
    NeuralNet *nn = nn_create(4, 0.01f, TENSOR_MSE_LOSS, NULL);

    nn_add_layer(nn, 0, linear_layer_create(3, 4));
    nn_add_layer(nn, 1, activation_layer_create(TANH));
    nn_add_layer(nn, 2, linear_layer_create(4, 2));
    nn_add_layer(nn, 3, activation_layer_create(SIGMOID));

    return nn;
}

// Set up gradients for testing (simulate a backward pass)
static void set_test_gradients(NeuralNet *nn) {
    for (int i = 0; i < nn->num_layers; i++) {
        Layer *layer = nn->layers[i];

        // Only set gradients for layers with parameters (Linear layers)
        if (layer->type == LAYER_LINEAR) {
            LinearLayer *ll = (LinearLayer *)layer->layer;

            // Set some non-zero gradients
            for (int row = 0; row < ll->grad_weights->shape[0]; row++) {
                for (int col = 0; col < ll->grad_weights->shape[1]; col++) {
                    tensor_set2d(ll->grad_weights, row, col, 0.1f * (row + col + 1));
                }
            }

            for (int j = 0; j < ll->grad_biases->size; j++) {
                ll->grad_biases->data[j] = 0.1f * (j + 1);
            }
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
    assert(opt->v == NULL);

    optimizer_free(opt);
    TEST_PASSED;
}

void test_optimizer_create_momentum() {
    Optimizer *opt = optimizer_create_momentum(0.01f, 0.9f);

    assert(opt != NULL);
    assert(opt->type == OPTIMIZER_MOMENTUM);
    assert(float_equals(opt->learning_rate, 0.01f));
    assert(float_equals(opt->beta, 0.9f));
    // v should be NULL until init
    assert(opt->v == NULL);

    optimizer_free(opt);
    TEST_PASSED;
}

void test_optimizer_create_adam() {
    Optimizer *opt = optimizer_create_adam(0.001f, 0.9f, 0.999f, 1e-8f);

    assert(opt != NULL);
    assert(opt->type == OPTIMIZER_ADAM);
    assert(float_equals(opt->learning_rate, 0.001f));
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
    NeuralNet *nn = create_test_nn();
    Optimizer *opt = optimizer_create_sgd(0.01f);

    optimizer_init(opt, nn);

    // Network has 2 linear layers, each with 2 params (weights + biases) = 4 total
    assert(opt->num_params == 4);
    // SGD shouldn't allocate velocity arrays
    assert(opt->v == NULL);

    optimizer_free(opt);
    nn_free(nn);
    TEST_PASSED;
}

void test_optimizer_init_momentum() {
    NeuralNet *nn = create_test_nn();
    Optimizer *opt = optimizer_create_momentum(0.01f, 0.9f);

    optimizer_init(opt, nn);

    // Network has 2 linear layers, each with 2 params (weights + biases) = 4 total
    assert(opt->num_params == 4);
    assert(opt->v != NULL);

    // Check that all velocity tensors are allocated and initialized to zero
    int param_idx = 0;
    for (int i = 0; i < nn->num_layers; i++) {
        LayerParameters params = layer_get_parameters(nn->layers[i]);

        for (int j = 0; j < params.num_pairs; j++) {
            Tensor *v = opt->v[param_idx];
            Tensor *param = params.pairs[j].param;

            // Check velocity tensor has same shape as parameter
            assert(v != NULL);
            assert(v->size == param->size);

            // Check initialized to zero
            for (int k = 0; k < v->size; k++) {
                assert(float_equals(v->data[k], 0.0f));
            }

            param_idx++;
        }

        layer_parameters_free(&params);
    }

    optimizer_free(opt);
    nn_free(nn);
    TEST_PASSED;
}

void test_optimizer_init_adam() {
    NeuralNet *nn = create_test_nn();
    Optimizer *opt = optimizer_create_adam(0.001f, 0.9f, 0.999f, 1e-8f);

    optimizer_init(opt, nn);

    // Network has 2 linear layers, each with 2 params (weights + biases) = 4 total
    assert(opt->num_params == 4);
    assert(opt->m != NULL);
    assert(opt->s != NULL);

    // Check that all moment tensors are allocated and initialized to zero
    int param_idx = 0;
    for (int i = 0; i < nn->num_layers; i++) {
        LayerParameters params = layer_get_parameters(nn->layers[i]);

        for (int j = 0; j < params.num_pairs; j++) {
            Tensor *m = opt->m[param_idx];
            Tensor *s = opt->s[param_idx];
            Tensor *param = params.pairs[j].param;

            // Check moment tensors have same shape as parameter
            assert(m != NULL);
            assert(s != NULL);
            assert(m->size == param->size);
            assert(s->size == param->size);

            // Check initialized to zero
            for (int k = 0; k < m->size; k++) {
                assert(float_equals(m->data[k], 0.0f));
                assert(float_equals(s->data[k], 0.0f));
            }

            param_idx++;
        }

        layer_parameters_free(&params);
    }

    optimizer_free(opt);
    nn_free(nn);
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
    NeuralNet *nn = create_test_nn();
    Optimizer *opt = optimizer_create_sgd(0.1f);
    optimizer_init(opt, nn);

    // First layer is linear (index 0)
    LinearLayer *layer_one = (LinearLayer *)(nn->layers[0]->layer);

    // Store original weights
    float original_weight = tensor_get2d(layer_one->weights, 0, 0);
    float original_bias = layer_one->biases->data[0];

    // Set gradients
    set_test_gradients(nn);
    float gradient_w = tensor_get2d(layer_one->grad_weights, 0, 0);
    float gradient_b = layer_one->grad_biases->data[0];

    // Step should perform: W = W - lr * grad_weights
    optimizer_step(opt, nn);

    float expected_weight = original_weight - 0.1f * gradient_w;
    float expected_bias = original_bias - 0.1f * gradient_b;

    assert(float_equals(tensor_get2d(layer_one->weights, 0, 0), expected_weight));
    assert(float_equals(layer_one->biases->data[0], expected_bias));

    optimizer_free(opt);
    nn_free(nn);
    TEST_PASSED;
}

void test_optimizer_step_sgd_zero_lr() {
    NeuralNet *nn = create_test_nn();
    Optimizer *opt = optimizer_create_sgd(0.0f);
    optimizer_init(opt, nn);

    LinearLayer *layer_one = (LinearLayer *)(nn->layers[0]->layer);

    float original_weight = tensor_get2d(layer_one->weights, 0, 0);
    set_test_gradients(nn);

    // With zero learning rate, weights shouldn't change
    optimizer_step(opt, nn);

    assert(float_equals(tensor_get2d(layer_one->weights, 0, 0), original_weight));

    optimizer_free(opt);
    nn_free(nn);
    TEST_PASSED;
}

// =============================================================================
// Momentum Optimizer Step Tests
// =============================================================================

void test_optimizer_step_momentum_first_step() {
    NeuralNet *nn = create_test_nn();
    Optimizer *opt = optimizer_create_momentum(0.1f, 0.9f);
    optimizer_init(opt, nn);

    LinearLayer *layer_one = (LinearLayer *)(nn->layers[0]->layer);

    float original_weight = tensor_get2d(layer_one->weights, 0, 0);
    float original_bias = layer_one->biases->data[0];

    set_test_gradients(nn);
    float gradient_w = tensor_get2d(layer_one->grad_weights, 0, 0);
    float gradient_b = layer_one->grad_biases->data[0];

    // First step: v = 0.9 * 0 + grad = grad, param = param - lr * v
    optimizer_step(opt, nn);

    // Check velocity was updated correctly
    // param_idx 0 = layer 0 weights, param_idx 1 = layer 0 biases
    assert(float_equals(tensor_get2d(opt->v[0], 0, 0), gradient_w));
    assert(float_equals(opt->v[1]->data[0], gradient_b));

    // Check weights were updated correctly
    float expected_weight = original_weight - 0.1f * gradient_w;
    float expected_bias = original_bias - 0.1f * gradient_b;

    assert(float_equals(tensor_get2d(layer_one->weights, 0, 0), expected_weight));
    assert(float_equals(layer_one->biases->data[0], expected_bias));

    optimizer_free(opt);
    nn_free(nn);
    TEST_PASSED;
}

void test_optimizer_step_momentum_second_step() {
    NeuralNet *nn = create_test_nn();
    Optimizer *opt = optimizer_create_momentum(0.1f, 0.9f);
    optimizer_init(opt, nn);

    LinearLayer *layer_one = (LinearLayer *)(nn->layers[0]->layer);

    set_test_gradients(nn);
    float gradient_w = tensor_get2d(layer_one->grad_weights, 0, 0);

    // First step
    optimizer_step(opt, nn);
    float weight_after_first = tensor_get2d(layer_one->weights, 0, 0);
    float velocity_after_first = tensor_get2d(opt->v[0], 0, 0); // param_idx 0 = weights

    // Second step with same gradients
    set_test_gradients(nn);
    optimizer_step(opt, nn);

    // v = 0.9 * velocity_after_first + gradient_w
    float expected_velocity = 0.9f * velocity_after_first + gradient_w;
    assert(float_equals(tensor_get2d(opt->v[0], 0, 0), expected_velocity));

    // W = weight_after_first - 0.1 * expected_velocity
    float expected_weight = weight_after_first - 0.1f * expected_velocity;
    assert(float_equals(tensor_get2d(layer_one->weights, 0, 0), expected_weight));

    optimizer_free(opt);
    nn_free(nn);
    TEST_PASSED;
}

void test_optimizer_step_momentum_accumulation() {
    NeuralNet *nn = create_test_nn();
    Optimizer *opt = optimizer_create_momentum(0.01f, 0.9f);
    optimizer_init(opt, nn);

    LinearLayer *layer_one = (LinearLayer *)(nn->layers[0]->layer);

    float original_weight = tensor_get2d(layer_one->weights, 0, 0);

    // Run multiple steps - velocity should accumulate
    for (int step = 0; step < 5; step++) {
        set_test_gradients(nn);
        optimizer_step(opt, nn);
    }

    // Weight should have moved more than a single SGD step would
    float final_weight = tensor_get2d(layer_one->weights, 0, 0);
    float gradient_w = 0.1f; // From set_test_gradients for (0,0)
    float single_sgd_step = original_weight - 0.01f * gradient_w;

    // Momentum should cause larger change
    assert(fabs(final_weight - original_weight) > fabs(single_sgd_step - original_weight));

    optimizer_free(opt);
    nn_free(nn);
    TEST_PASSED;
}

void test_optimizer_step_momentum_beta_zero() {
    NeuralNet *nn = create_test_nn();
    Optimizer *opt = optimizer_create_momentum(0.1f, 0.0f); // beta = 0 => SGD
    optimizer_init(opt, nn);

    LinearLayer *layer_one = (LinearLayer *)(nn->layers[0]->layer);

    float original_weight = tensor_get2d(layer_one->weights, 0, 0);
    set_test_gradients(nn);
    float gradient_w = tensor_get2d(layer_one->grad_weights, 0, 0);

    optimizer_step(opt, nn);

    // With beta=0, should behave like SGD
    float expected_weight = original_weight - 0.1f * gradient_w;
    assert(float_equals(tensor_get2d(layer_one->weights, 0, 0), expected_weight));

    optimizer_free(opt);
    nn_free(nn);
    TEST_PASSED;
}

// =============================================================================
// Adam Optimizer Step Tests
// =============================================================================

void test_optimizer_step_adam_first_step() {
    NeuralNet *nn = create_test_nn();
    Optimizer *opt = optimizer_create_adam(0.001f, 0.9f, 0.999f, 1e-8f);
    optimizer_init(opt, nn);

    LinearLayer *layer_one = (LinearLayer *)(nn->layers[0]->layer);

    float original_weight = tensor_get2d(layer_one->weights, 0, 0);
    set_test_gradients(nn);
    float gradient_w = tensor_get2d(layer_one->grad_weights, 0, 0);

    optimizer_step(opt, nn);

    // Check timestep was incremented
    assert(opt->timestep == 1);

    // First step: m = (1-β₁)·g, s = (1-β₂)·g²
    float expected_m = (1.0f - 0.9f) * gradient_w;
    float expected_s = (1.0f - 0.999f) * gradient_w * gradient_w;

    // Check moment estimates were updated
    // param_idx 0 = layer 0 weights
    assert(float_equals(tensor_get2d(opt->m[0], 0, 0), expected_m));
    assert(float_equals(tensor_get2d(opt->s[0], 0, 0), expected_s));

    // Check weight was updated with bias-corrected moments
    float bc1 = 1.0f - powf(0.9f, 1);
    float bc2 = 1.0f - powf(0.999f, 1);
    float m_hat = expected_m / bc1;
    float s_hat = expected_s / bc2;
    float expected_weight = original_weight - 0.001f * m_hat / (sqrtf(s_hat) + 1e-8f);

    assert(float_equals(tensor_get2d(layer_one->weights, 0, 0), expected_weight));

    optimizer_free(opt);
    nn_free(nn);
    TEST_PASSED;
}

void test_optimizer_step_adam_second_step() {
    NeuralNet *nn = create_test_nn();
    Optimizer *opt = optimizer_create_adam(0.001f, 0.9f, 0.999f, 1e-8f);
    optimizer_init(opt, nn);

    LinearLayer *layer_one = (LinearLayer *)(nn->layers[0]->layer);

    set_test_gradients(nn);
    float gradient_w = tensor_get2d(layer_one->grad_weights, 0, 0);

    // First step
    optimizer_step(opt, nn);
    float m_after_first = tensor_get2d(opt->m[0], 0, 0);
    float s_after_first = tensor_get2d(opt->s[0], 0, 0);
    float weight_after_first = tensor_get2d(layer_one->weights, 0, 0);

    // Second step with same gradients
    set_test_gradients(nn);
    optimizer_step(opt, nn);

    // Check timestep
    assert(opt->timestep == 2);

    // m = β₁·m_prev + (1-β₁)·g
    float expected_m = 0.9f * m_after_first + (1.0f - 0.9f) * gradient_w;
    assert(float_equals(tensor_get2d(opt->m[0], 0, 0), expected_m));

    // s = β₂·s_prev + (1-β₂)·g²
    float expected_s = 0.999f * s_after_first + (1.0f - 0.999f) * gradient_w * gradient_w;
    assert(float_equals(tensor_get2d(opt->s[0], 0, 0), expected_s));

    // Weight should have changed from first step
    assert(!float_equals(tensor_get2d(layer_one->weights, 0, 0), weight_after_first));

    optimizer_free(opt);
    nn_free(nn);
    TEST_PASSED;
}

void test_optimizer_step_adam_bias_correction() {
    NeuralNet *nn = create_test_nn();
    Optimizer *opt = optimizer_create_adam(0.001f, 0.9f, 0.999f, 1e-8f);
    optimizer_init(opt, nn);

    LinearLayer *layer_one = (LinearLayer *)(nn->layers[0]->layer);

    set_test_gradients(nn);
    float gradient_w = tensor_get2d(layer_one->grad_weights, 0, 0);
    float original_weight = tensor_get2d(layer_one->weights, 0, 0);

    optimizer_step(opt, nn);

    // Manually compute what the update should be with bias correction
    float m = (1.0f - 0.9f) * gradient_w;
    float s = (1.0f - 0.999f) * gradient_w * gradient_w;

    // Bias correction factors for t=1
    float bc1 = 1.0f - powf(0.9f, 1);
    float bc2 = 1.0f - powf(0.999f, 1);

    float m_hat = m / bc1;
    float s_hat = s / bc2;

    float expected_update = 0.001f * m_hat / (sqrtf(s_hat) + 1e-8f);
    float expected_weight = original_weight - expected_update;

    assert(float_equals(tensor_get2d(layer_one->weights, 0, 0), expected_weight));

    optimizer_free(opt);
    nn_free(nn);
    TEST_PASSED;
}

void test_optimizer_step_adam_all_parameters() {
    NeuralNet *nn = create_test_nn();
    Optimizer *opt = optimizer_create_adam(0.001f, 0.9f, 0.999f, 1e-8f);
    optimizer_init(opt, nn);

    // Store original weights and biases for both linear layers
    LinearLayer *layer1 = (LinearLayer *)(nn->layers[0]->layer);
    LinearLayer *layer2 = (LinearLayer *)(nn->layers[2]->layer);

    float original_w1 = tensor_get2d(layer1->weights, 0, 0);
    float original_b1 = layer1->biases->data[0];
    float original_w2 = tensor_get2d(layer2->weights, 0, 0);
    float original_b2 = layer2->biases->data[0];

    set_test_gradients(nn);
    optimizer_step(opt, nn);

    // Check that all 4 parameters were updated (2 layers × 2 params each)
    assert(!float_equals(tensor_get2d(layer1->weights, 0, 0), original_w1));
    assert(!float_equals(layer1->biases->data[0], original_b1));
    assert(!float_equals(tensor_get2d(layer2->weights, 0, 0), original_w2));
    assert(!float_equals(layer2->biases->data[0], original_b2));

    // Check that all moment estimates are non-zero
    for (int i = 0; i < opt->num_params; i++) {
        assert(opt->m[i] != NULL);
        assert(opt->s[i] != NULL);
        // At least one element should be non-zero (since we set gradients)
        int has_nonzero = 0;
        for (int j = 0; j < opt->m[i]->size; j++) {
            if (!float_equals(opt->m[i]->data[j], 0.0f)) {
                has_nonzero = 1;
                break;
            }
        }
        assert(has_nonzero);
    }

    optimizer_free(opt);
    nn_free(nn);
    TEST_PASSED;
}

// =============================================================================
// Multi-layer Tests
// =============================================================================

void test_optimizer_step_all_layers() {
    NeuralNet *nn = create_test_nn();
    Optimizer *opt = optimizer_create_momentum(0.1f, 0.9f);
    optimizer_init(opt, nn);

    // Store original weights for linear layers
    LinearLayer *layer1 = (LinearLayer *)(nn->layers[0]->layer);
    LinearLayer *layer2 = (LinearLayer *)(nn->layers[2]->layer);
    float original_weight1 = tensor_get2d(layer1->weights, 0, 0);
    float original_weight2 = tensor_get2d(layer2->weights, 0, 0);

    set_test_gradients(nn);
    optimizer_step(opt, nn);

    // Check that both linear layers were updated
    float new_weight1 = tensor_get2d(layer1->weights, 0, 0);
    float new_weight2 = tensor_get2d(layer2->weights, 0, 0);

    assert(!float_equals(new_weight1, original_weight1));
    assert(!float_equals(new_weight2, original_weight2));

    optimizer_free(opt);
    nn_free(nn);
    TEST_PASSED;
}

// =============================================================================
// Edge Cases Tests
// =============================================================================

void test_optimizer_step_zero_gradients() {
    NeuralNet *nn = create_test_nn();
    Optimizer *opt = optimizer_create_sgd(0.1f);
    optimizer_init(opt, nn);

    LinearLayer *layer_one = (LinearLayer *)(nn->layers[0]->layer);

    float original_weight = tensor_get2d(layer_one->weights, 0, 0);

    // Don't set gradients - they should be zero
    nn_zero_gradients(nn);
    optimizer_step(opt, nn);

    // Weights shouldn't change with zero gradients
    assert(float_equals(tensor_get2d(layer_one->weights, 0, 0), original_weight));

    optimizer_free(opt);
    nn_free(nn);
    TEST_PASSED;
}

void test_optimizer_step_negative_gradients() {
    NeuralNet *nn = create_test_nn();
    Optimizer *opt = optimizer_create_sgd(0.1f);
    optimizer_init(opt, nn);

    LinearLayer *layer_one = (LinearLayer *)(nn->layers[0]->layer);

    float original_weight = tensor_get2d(layer_one->weights, 0, 0);

    // Set negative gradients
    tensor_set2d(layer_one->grad_weights, 0, 0, -1.0f);

    optimizer_step(opt, nn);

    // W = W - lr * (-1.0) = W + lr
    float expected = original_weight + 0.1f;
    assert(float_equals(tensor_get2d(layer_one->weights, 0, 0), expected));

    optimizer_free(opt);
    nn_free(nn);
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
    test_optimizer_init_adam();

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

    printf("\n=== Adam Optimizer Step Tests ===\n");
    test_optimizer_step_adam_first_step();
    test_optimizer_step_adam_second_step();
    test_optimizer_step_adam_bias_correction();
    test_optimizer_step_adam_all_parameters();

    printf("\n=== Multi-layer Optimizer Tests ===\n");
    test_optimizer_step_all_layers();

    printf("\n=== Optimizer Edge Cases Tests ===\n");
    test_optimizer_step_zero_gradients();
    test_optimizer_step_negative_gradients();

    printf("\n=== All Optimizer Tests Passed! ===\n");
}
