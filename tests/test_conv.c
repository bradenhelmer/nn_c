/*
 * test_conv.c - Tests for convolutional layer operations
 *
 * Uses Sobel edge detector as a worked example to verify forward and backward pass.
 */

#include "../src/nn/conv_layer.h"
#include "../src/tensor/tensor.h"
#include "../src/utils/utils.h"
#include "test_runner.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// =============================================================================
// Sobel Edge Detector Test (Forward Pass)
// =============================================================================

void test_conv_forward_sobel() {
    // Setup:
    // Input X:  [1, 4, 4]  (1 channel, 4×4)
    // Kernel K: [1, 1, 3, 3]  (1 output channel, 1 input channel, 3×3)
    // Stride: 1, Padding: 0
    // Output Y: [1, 2, 2]

    ConvLayer *layer = conv_layer_create(1, 1, 3, 1, 0);

    // Input X[0]:
    // [1, 2, 3, 4]
    // [5, 6, 7, 8]
    // [9, 10, 11, 12]
    // [13, 14, 15, 16]
    Tensor *input = tensor_create(3, (int[]){1, 4, 4});
    float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    for (int i = 0; i < 16; i++) {
        input->data[i] = input_data[i];
    }

    // Sobel kernel for vertical edges:
    // [1, 0, -1]
    // [2, 0, -2]
    // [1, 0, -1]
    float sobel_kernel[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    for (int i = 0; i < 9; i++) {
        layer->kernels->data[i] = sobel_kernel[i];
    }

    // Set bias to 0
    layer->biases->data[0] = 0.0f;

    // Run forward pass
    Tensor *output = conv_layer_forward(layer, input);

    // Expected output Y[0]:
    // [-8, -8]
    // [-8, -8]
    assert(output->ndim == 3);
    assert(output->shape[0] == 1);
    assert(output->shape[1] == 2);
    assert(output->shape[2] == 2);

    assert(float_equals(tensor_get3d(output, 0, 0, 0), -8.0f));
    assert(float_equals(tensor_get3d(output, 0, 0, 1), -8.0f));
    assert(float_equals(tensor_get3d(output, 0, 1, 0), -8.0f));
    assert(float_equals(tensor_get3d(output, 0, 1, 1), -8.0f));

    tensor_free(input);
    conv_layer_free(layer);
    TEST_PASSED;
}

// =============================================================================
// Sobel Edge Detector Test (Backward Pass)
// =============================================================================

void test_conv_backward_sobel() {
    // Same setup as forward
    ConvLayer *layer = conv_layer_create(1, 1, 3, 1, 0);

    // Input X[0]:
    Tensor *input = tensor_create(3, (int[]){1, 4, 4});
    float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
    for (int i = 0; i < 16; i++) {
        input->data[i] = input_data[i];
    }

    // Sobel kernel
    float sobel_kernel[] = {1, 0, -1, 2, 0, -2, 1, 0, -1};
    for (int i = 0; i < 9; i++) {
        layer->kernels->data[i] = sobel_kernel[i];
    }
    layer->biases->data[0] = 0.0f;

    // Run forward pass first (needed to cache input)
    conv_layer_forward(layer, input);

    // Upstream gradient δ[0]:
    // [1, 0]
    // [0, 1]
    Tensor *upstream_grad = tensor_create(3, (int[]){1, 2, 2});
    tensor_set3d(upstream_grad, 0, 0, 0, 1.0f);
    tensor_set3d(upstream_grad, 0, 0, 1, 0.0f);
    tensor_set3d(upstream_grad, 0, 1, 0, 0.0f);
    tensor_set3d(upstream_grad, 0, 1, 1, 1.0f);

    // Run backward pass
    Tensor *dX = conv_layer_backward(layer, upstream_grad);

    // Check bias gradient: db[0] = 1 + 0 + 0 + 1 = 2
    assert(float_equals(layer->d_biases->data[0], 2.0f));

    // Check kernel gradient dK[0,0,0,0] (top-left kernel element):
    // dK[0,0,0,0] = δ[0,0,0] * X[0,0,0] + δ[0,1,1] * X[0,1,1]
    //            = 1 * 1 + 1 * 6 = 7
    assert(float_equals(tensor_get4d(layer->d_kernels, 0, 0, 0, 0), 7.0f));

    // Check input gradient dX[0,0,0] (top-left input element):
    // Only Y[0,0,0] depends on X[0,0,0], through K[0,0,0,0]:
    // dX[0,0,0] = δ[0,0,0] * K[0,0,0,0] = 1 * 1 = 1
    assert(float_equals(tensor_get3d(dX, 0, 0, 0), 1.0f));

    tensor_free(input);
    tensor_free(upstream_grad);
    tensor_free(dX);
    conv_layer_free(layer);
    TEST_PASSED;
}

// =============================================================================
// Conv Layer Creation Test
// =============================================================================

void test_conv_layer_create() {
    ConvLayer *layer = conv_layer_create(3, 16, 3, 1, 1);

    assert(layer != NULL);
    assert(layer->in_channels == 3);
    assert(layer->out_channels == 16);
    assert(layer->kernel_size == 3);
    assert(layer->stride == 1);
    assert(layer->padding == 1);

    // Kernels should be [16, 3, 3, 3]
    assert(layer->kernels->ndim == 4);
    assert(layer->kernels->shape[0] == 16);
    assert(layer->kernels->shape[1] == 3);
    assert(layer->kernels->shape[2] == 3);
    assert(layer->kernels->shape[3] == 3);
    assert(layer->kernels->size == 16 * 3 * 3 * 3);

    // Biases should be [16]
    assert(layer->biases->ndim == 1);
    assert(layer->biases->shape[0] == 16);

    conv_layer_free(layer);
    TEST_PASSED;
}

// =============================================================================
// Forward Output Dimensions Test
// =============================================================================

void test_conv_forward_dimensions() {
    // Test that output dimensions are computed correctly
    // H_out = (H_in - K + 2*P) / S + 1

    // Case 1: 28x28 input, 3x3 kernel, stride 1, padding 1 -> 28x28 output
    ConvLayer *layer1 = conv_layer_create(1, 1, 3, 1, 1);
    Tensor *input1 = tensor_zeros(3, (int[]){1, 28, 28});
    Tensor *out1 = conv_layer_forward(layer1, input1);
    assert(out1->shape[1] == 28);
    assert(out1->shape[2] == 28);
    tensor_free(input1);
    conv_layer_free(layer1);

    // Case 2: 28x28 input, 3x3 kernel, stride 1, padding 0 -> 26x26 output
    ConvLayer *layer2 = conv_layer_create(1, 1, 3, 1, 0);
    Tensor *input2 = tensor_zeros(3, (int[]){1, 28, 28});
    Tensor *out2 = conv_layer_forward(layer2, input2);
    assert(out2->shape[1] == 26);
    assert(out2->shape[2] == 26);
    tensor_free(input2);
    conv_layer_free(layer2);

    // Case 3: 28x28 input, 5x5 kernel, stride 2, padding 0 -> 12x12 output
    ConvLayer *layer3 = conv_layer_create(1, 1, 5, 2, 0);
    Tensor *input3 = tensor_zeros(3, (int[]){1, 28, 28});
    Tensor *out3 = conv_layer_forward(layer3, input3);
    assert(out3->shape[1] == 12);
    assert(out3->shape[2] == 12);
    tensor_free(input3);
    conv_layer_free(layer3);

    TEST_PASSED;
}

// =============================================================================
// Forward with Bias Test
// =============================================================================

void test_conv_forward_with_bias() {
    ConvLayer *layer = conv_layer_create(1, 1, 3, 1, 0);

    // Simple input of all 1s
    Tensor *input = tensor_zeros(3, (int[]){1, 3, 3});
    tensor_fill(input, 1.0f);

    // Kernel of all 1s (sum = 9)
    tensor_fill(layer->kernels, 1.0f);

    // Bias of 5
    layer->biases->data[0] = 5.0f;

    Tensor *output = conv_layer_forward(layer, input);

    // Output should be 3x3 kernel sum (9) + bias (5) = 14
    assert(output->shape[1] == 1);
    assert(output->shape[2] == 1);
    assert(float_equals(tensor_get3d(output, 0, 0, 0), 14.0f));

    tensor_free(input);
    conv_layer_free(layer);
    TEST_PASSED;
}

// =============================================================================
// Test Runner
// =============================================================================

void run_conv_tests(void) {
    printf("\n=== Convolutional Layer Tests ===\n");
    test_conv_layer_create();
    test_conv_forward_dimensions();
    test_conv_forward_with_bias();
    test_conv_forward_sobel();
    test_conv_backward_sobel();

    printf("\n=== All Conv Tests Passed! ===\n");
}
