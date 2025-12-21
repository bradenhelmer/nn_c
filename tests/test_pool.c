/*
 * test_pool.c - Tests for max pooling layer operations
 *
 * Uses a worked example to verify forward and backward pass.
 */

#include "../src/nn/layer.h"
#include "../src/tensor/tensor.h"
#include "../src/utils/utils.h"
#include "test_runner.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// =============================================================================
// Max Pool Forward Pass Test
// =============================================================================

void test_maxpool_forward() {
    // Setup:
    // Input X:  [1, 4, 4]  (1 channel, 4×4)
    // Pool size: 2, Stride: 2
    // Output Y: [1, 2, 2]

    MaxPoolLayer *layer = maxpool_create(2, 2);

    // Input X[0]:
    // [1, 3, 2, 4]
    // [5, 6, 1, 2]
    // [7, 2, 3, 1]
    // [4, 8, 5, 6]
    Tensor *input = tensor_create(3, (int[]){1, 4, 4});
    float input_data[] = {1, 3, 2, 4, 5, 6, 1, 2, 7, 2, 3, 1, 4, 8, 5, 6};
    for (int i = 0; i < 16; i++) {
        input->data[i] = input_data[i];
    }

    // Run forward pass
    Tensor *output = maxpool_forward(layer, input);

    // Expected output Y[0]:
    // [6, 4]
    // [8, 6]
    assert(output->ndim == 3);
    assert(output->shape[0] == 1);
    assert(output->shape[1] == 2);
    assert(output->shape[2] == 2);

    assert(float_equals(tensor_get3d(output, 0, 0, 0), 6.0f));
    assert(float_equals(tensor_get3d(output, 0, 0, 1), 4.0f));
    assert(float_equals(tensor_get3d(output, 0, 1, 0), 8.0f));
    assert(float_equals(tensor_get3d(output, 0, 1, 1), 6.0f));

    // Verify max_indices (flat window indices)
    // Window (0,0): max at (1,1) -> flat_idx = 1*2 + 1 = 3
    // Window (0,1): max at (0,1) -> flat_idx = 0*2 + 1 = 1
    // Window (1,0): max at (1,1) -> flat_idx = 1*2 + 1 = 3
    // Window (1,1): max at (1,1) -> flat_idx = 1*2 + 1 = 3
    assert(layer->max_indices[0] == 3);
    assert(layer->max_indices[1] == 1);
    assert(layer->max_indices[2] == 3);
    assert(layer->max_indices[3] == 3);

    tensor_free(input);
    tensor_free(output);
    maxpool_free(layer);
    TEST_PASSED;
}

// =============================================================================
// Max Pool Backward Pass Test
// =============================================================================

void test_maxpool_backward() {
    // Same setup as forward
    MaxPoolLayer *layer = maxpool_create(2, 2);

    // Input X[0]:
    // [1, 3, 2, 4]
    // [5, 6, 1, 2]
    // [7, 2, 3, 1]
    // [4, 8, 5, 6]
    Tensor *input = tensor_create(3, (int[]){1, 4, 4});
    float input_data[] = {1, 3, 2, 4, 5, 6, 1, 2, 7, 2, 3, 1, 4, 8, 5, 6};
    for (int i = 0; i < 16; i++) {
        input->data[i] = input_data[i];
    }

    // Run forward pass first (needed to cache max_indices)
    Tensor *output = maxpool_forward(layer, input);

    // Upstream gradient δ[0]:
    // [1, 2]
    // [3, 4]
    Tensor *upstream_grad = tensor_create(3, (int[]){1, 2, 2});
    tensor_set3d(upstream_grad, 0, 0, 0, 1.0f);
    tensor_set3d(upstream_grad, 0, 0, 1, 2.0f);
    tensor_set3d(upstream_grad, 0, 1, 0, 3.0f);
    tensor_set3d(upstream_grad, 0, 1, 1, 4.0f);

    // Run backward pass
    Tensor *dX = maxpool_backward(layer, upstream_grad);

    // Expected dX[0]:
    // [0, 0, 0, 2]
    // [0, 1, 0, 0]
    // [0, 0, 0, 0]
    // [0, 3, 0, 4]
    //
    // Gradient routing:
    // δ=1 at (0,0) -> max was at window pos (1,1) -> X[0,1,1]
    // δ=2 at (0,1) -> max was at window pos (0,1) -> X[0,0,3]
    // δ=3 at (1,0) -> max was at window pos (1,1) -> X[0,3,1]
    // δ=4 at (1,1) -> max was at window pos (1,1) -> X[0,3,3]

    assert(dX->ndim == 3);
    assert(dX->shape[0] == 1);
    assert(dX->shape[1] == 4);
    assert(dX->shape[2] == 4);

    // Row 0: [0, 0, 0, 2]
    assert(float_equals(tensor_get3d(dX, 0, 0, 0), 0.0f));
    assert(float_equals(tensor_get3d(dX, 0, 0, 1), 0.0f));
    assert(float_equals(tensor_get3d(dX, 0, 0, 2), 0.0f));
    assert(float_equals(tensor_get3d(dX, 0, 0, 3), 2.0f));

    // Row 1: [0, 1, 0, 0]
    assert(float_equals(tensor_get3d(dX, 0, 1, 0), 0.0f));
    assert(float_equals(tensor_get3d(dX, 0, 1, 1), 1.0f));
    assert(float_equals(tensor_get3d(dX, 0, 1, 2), 0.0f));
    assert(float_equals(tensor_get3d(dX, 0, 1, 3), 0.0f));

    // Row 2: [0, 0, 0, 0]
    assert(float_equals(tensor_get3d(dX, 0, 2, 0), 0.0f));
    assert(float_equals(tensor_get3d(dX, 0, 2, 1), 0.0f));
    assert(float_equals(tensor_get3d(dX, 0, 2, 2), 0.0f));
    assert(float_equals(tensor_get3d(dX, 0, 2, 3), 0.0f));

    // Row 3: [0, 3, 0, 4]
    assert(float_equals(tensor_get3d(dX, 0, 3, 0), 0.0f));
    assert(float_equals(tensor_get3d(dX, 0, 3, 1), 3.0f));
    assert(float_equals(tensor_get3d(dX, 0, 3, 2), 0.0f));
    assert(float_equals(tensor_get3d(dX, 0, 3, 3), 4.0f));

    tensor_free(input);
    tensor_free(output);
    tensor_free(upstream_grad);
    tensor_free(dX);
    maxpool_free(layer);
    TEST_PASSED;
}

// =============================================================================
// Max Pool Layer Creation Test
// =============================================================================

void test_maxpool_create() {
    MaxPoolLayer *layer = maxpool_create(2, 2);

    assert(layer != NULL);
    assert(layer->pool_size == 2);
    assert(layer->stride == 2);
    assert(layer->max_indices == NULL);

    maxpool_free(layer);
    TEST_PASSED;
}

// =============================================================================
// Forward Output Dimensions Test
// =============================================================================

void test_maxpool_forward_dimensions() {
    // Test that output dimensions are computed correctly
    // H_out = (H_in - pool_size) / stride + 1

    // Case 1: 4x4 input, pool_size 2, stride 2 -> 2x2 output
    MaxPoolLayer *layer1 = maxpool_create(2, 2);
    Tensor *input1 = tensor_zeros(3, (int[]){1, 4, 4});
    Tensor *out1 = maxpool_forward(layer1, input1);
    assert(out1->shape[1] == 2);
    assert(out1->shape[2] == 2);
    tensor_free(input1);
    tensor_free(out1);
    maxpool_free(layer1);

    // Case 2: 6x6 input, pool_size 2, stride 2 -> 3x3 output
    MaxPoolLayer *layer2 = maxpool_create(2, 2);
    Tensor *input2 = tensor_zeros(3, (int[]){1, 6, 6});
    Tensor *out2 = maxpool_forward(layer2, input2);
    assert(out2->shape[1] == 3);
    assert(out2->shape[2] == 3);
    tensor_free(input2);
    tensor_free(out2);
    maxpool_free(layer2);

    // Case 3: 8x8 input, pool_size 3, stride 2 -> 3x3 output
    MaxPoolLayer *layer3 = maxpool_create(3, 2);
    Tensor *input3 = tensor_zeros(3, (int[]){1, 8, 8});
    Tensor *out3 = maxpool_forward(layer3, input3);
    assert(out3->shape[1] == 3);
    assert(out3->shape[2] == 3);
    tensor_free(input3);
    tensor_free(out3);
    maxpool_free(layer3);

    TEST_PASSED;
}

// =============================================================================
// Test Runner
// =============================================================================

void run_pool_tests(void) {
    printf("\n=== Max Pooling Layer Tests ===\n");
    test_maxpool_create();
    test_maxpool_forward_dimensions();
    test_maxpool_forward();
    test_maxpool_backward();

    printf("\n=== All Pool Tests Passed! ===\n");
}
