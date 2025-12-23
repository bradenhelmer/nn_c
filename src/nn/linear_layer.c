/*
 * linear_layer.c - Linear layer implementations.
 *
 */
#include "../utils/utils.h"
#include "layer.h"
#include <math.h>
#include <stdlib.h>

Layer *linear_layer_create(int input_size, int output_size) {
    LinearLayer *layer = (LinearLayer *)malloc(sizeof(LinearLayer));

    layer->input_size = input_size;
    layer->output_size = output_size;

    layer->weights = tensor_create2d(output_size, input_size);
    layer->biases = tensor_create1d(output_size);

    layer->output = tensor_create1d(output_size);
    layer->input = tensor_create1d(input_size);

    layer->grad_weights = tensor_create2d(output_size, input_size);
    layer->grad_biases = tensor_create1d(output_size);

    linear_layer_init_xavier(layer);

    return layer_create(LAYER_LINEAR, (void *)layer);
}

void linear_layer_free(LinearLayer *layer) {
    tensor_free(layer->weights);
    tensor_free(layer->biases);

    tensor_free(layer->output);
    tensor_free(layer->input);

    tensor_free(layer->grad_weights);
    tensor_free(layer->grad_biases);

    free(layer);
}

static void _layer_init(LinearLayer *layer, float standard) {
    float min = -standard, max = standard;
    int rows = layer->weights->shape[0];
    int cols = layer->weights->shape[1];
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            tensor_set2d(layer->weights, i, j, rand_rangef(min, max));
        }
    }
}

void linear_layer_init_xavier(LinearLayer *layer) {
    _layer_init(layer, sqrtf(2.f / (layer->input_size + layer->output_size)));
}
void linear_layer_init_he(LinearLayer *layer) {
    _layer_init(layer, sqrtf(2.f / layer->input_size));
}

// Forward/backward
void linear_layer_forward(LinearLayer *layer, const Tensor *input) {
    // Cache input for weight gradient calculation.
    tensor_copy(layer->input, input);

    // output = Wx + b
    tensor_matvec_mul(layer->output, layer->weights, input);
    tensor_add(layer->output, layer->output, layer->biases);
}

Tensor *linear_layer_backward(LinearLayer *layer, const Tensor *upstream_grad) {

    // 1. Weights -> grad_weights = upstream_grad ⊗ input^T (outer product) - accumulate
    Tensor *grad_sample = tensor_create2d(layer->output_size, layer->input_size);
    tensor_outer_product(grad_sample, upstream_grad, layer->input);
    tensor_add(layer->grad_weights, layer->grad_weights, grad_sample);
    tensor_free(grad_sample);

    // 2. Bias -> grad_biases = upstream_grad - accumulate
    tensor_add(layer->grad_biases, layer->grad_biases, upstream_grad);

    // 3. Input Gradient -> dL/dinput = W^T * upstream_grad (return to previous layer)
    Tensor *grad_input = tensor_create1d(layer->input_size);
    tensor_matvec_mul_transpose(grad_input, layer->weights, upstream_grad);

    return grad_input;
}
