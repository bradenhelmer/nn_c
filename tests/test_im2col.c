/*
 * test_im2col.c - Tests for convolutional layer im2col operations.
 */

#include "nn/layer.h"
#include "tensor/tensor.h"
#include "test_runner.h"
#include "utils/utils.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>

void test_im2col() {

    Tensor *X = tensor_create3d(1, 2, 2);
    X->data[0] = 1.0f;
    X->data[1] = 2.0f;
    X->data[2] = 3.0f;
    X->data[3] = 4.0f;

    Layer *l = conv2d_layer_create(1, 1, 3, 1, 1);
    Conv2DLayer *layer = l->layer;

    Tensor *X_pad = tensor_pad2d(X, 1);
    Tensor *converted = conv2d_im2col(layer, X_pad);

    assert(float_equals(converted->data[0], 0.0f));
    assert(float_equals(converted->data[1], 0.0f));
    assert(float_equals(converted->data[2], 0.0f));
    assert(float_equals(converted->data[3], 1.0f));
    assert(float_equals(converted->data[4], 0.0f));
    assert(float_equals(converted->data[5], 0.0f));
    assert(float_equals(converted->data[6], 1.0f));
    assert(float_equals(converted->data[7], 2.0f));
    assert(float_equals(converted->data[8], 0.0f));
    assert(float_equals(converted->data[9], 0.0f));
    assert(float_equals(converted->data[10], 2.0f));
    assert(float_equals(converted->data[11], 0.0f));
    assert(float_equals(converted->data[12], 0.0f));
    assert(float_equals(converted->data[13], 1.0f));
    assert(float_equals(converted->data[14], 0.0f));
    assert(float_equals(converted->data[15], 3.0f));
    assert(float_equals(converted->data[16], 1.0f));
    assert(float_equals(converted->data[17], 2.0f));
    assert(float_equals(converted->data[18], 3.0f));
    assert(float_equals(converted->data[19], 4.0f));
    assert(float_equals(converted->data[20], 2.0f));
    assert(float_equals(converted->data[21], 0.0f));
    assert(float_equals(converted->data[22], 4.0f));
    assert(float_equals(converted->data[23], 0.0f));
    assert(float_equals(converted->data[24], 0.0f));
    assert(float_equals(converted->data[25], 3.0f));
    assert(float_equals(converted->data[26], 0.0f));
    assert(float_equals(converted->data[27], 0.0f));
    assert(float_equals(converted->data[28], 3.0f));
    assert(float_equals(converted->data[29], 4.0f));
    assert(float_equals(converted->data[30], 0.0f));
    assert(float_equals(converted->data[31], 0.0f));
    assert(float_equals(converted->data[32], 4.0f));
    assert(float_equals(converted->data[33], 0.0f));
    assert(float_equals(converted->data[34], 0.0f));
    assert(float_equals(converted->data[35], 0.0f));

    tensor_free(X);
    tensor_free(X_pad);
    tensor_free(converted);

    TEST_PASSED;
}

void test_conv_forward_im2col() {
    // Test that im2col forward pass produces same result as regular forward pass
    // Setup: Small 2-channel input with 2 output channels, 3x3 kernel
    Layer *l = conv2d_layer_create(2, 2, 3, 1, 1);
    Conv2DLayer *layer = l->layer;

    // Create input [2, 4, 4] (2 channels, 4x4)
    Tensor *input = tensor_create3d(2, 4, 4);
    for (int i = 0; i < input->size; i++) {
        input->data[i] = (float)(i + 1);
    }

    // Set known weights and biases
    for (int i = 0; i < layer->weights->size; i++) {
        layer->weights->data[i] = 0.1f * (i % 9 - 4); // Small values centered around 0
    }
    layer->biases->data[0] = 0.5f;
    layer->biases->data[1] = -0.5f;

    // Run regular forward pass
    Tensor *output_regular = conv2d_layer_forward(layer, input);
    Tensor *output_regular_copy = tensor_clone(output_regular);

    // Reset layer state
    tensor_free(layer->input);
    tensor_free(layer->output);
    layer->input = NULL;
    layer->output = NULL;

    // Run im2col forward pass
    Tensor *output_im2col = conv_layer_forward_im2col(layer, input);

    // Verify shapes match
    assert(output_regular_copy->ndim == output_im2col->ndim);
    assert(output_regular_copy->shape[0] == output_im2col->shape[0]);
    assert(output_regular_copy->shape[1] == output_im2col->shape[1]);
    assert(output_regular_copy->shape[2] == output_im2col->shape[2]);

    // Use slightly larger tolerance for numerical comparisons
    // Im2col uses matrix multiplication which accumulates in different order
    float tolerance = 1e-4f;
    for (int i = 0; i < output_regular_copy->size; i++) {
        float diff = fabsf(output_regular_copy->data[i] - output_im2col->data[i]);
        if (diff > tolerance) {
            fprintf(stderr, "TEST: Mismatch at index %d: regular=%.6f, im2col=%.6f, diff=%.8f\n", i,
                    output_regular_copy->data[i], output_im2col->data[i], diff);
            assert(0);
        }
    }

    tensor_free(input);
    tensor_free(output_regular_copy);
    layer_free(l);
    TEST_PASSED;
}

void run_im2col_tests(void) {
    test_im2col();
    test_conv_forward_im2col();
}
