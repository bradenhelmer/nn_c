/*
 * linear_layer.c - Linear layer implementations.
 *
 */
#include "../utils/utils.h"
#include "layer.h"
#include "tensor/tensor.h"
#include <math.h>
#include <stdlib.h>

Layer *linear_layer_create(int input_size, int output_size) {
    LinearLayer *layer = (LinearLayer *)malloc(sizeof(LinearLayer));

    layer->input_size = input_size;
    layer->output_size = output_size;

    layer->weights = tensor_create2d(output_size, input_size);
    layer->biases = tensor_create1d(output_size);

    layer->input = NULL;
    layer->output = NULL;

    layer->grad_weights = tensor_create2d(output_size, input_size);
    layer->grad_biases = tensor_create1d(output_size);

    linear_layer_init_xavier(layer);

    return layer_create(LAYER_LINEAR, (void *)layer);
}

void linear_layer_free(LinearLayer *layer) {
    tensor_free(layer->weights);
    tensor_free(layer->biases);

    if (layer->output != NULL) {
        tensor_free(layer->output);
    }
    if (layer->input != NULL) {
        tensor_free(layer->input);
    }

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
Tensor *linear_layer_forward(LinearLayer *layer, const Tensor *input) {
    // Cache input for weight gradient calculation.
    if (layer->input != NULL) {
        tensor_free(layer->input);
    }
    layer->input = tensor_clone(input);

    // output = Wx + b
    Tensor *Y = tensor_create1d(layer->output_size);
    tensor_matvec_mul(Y, layer->weights, input);
    tensor_add(Y, Y, layer->biases);

    if (layer->output != NULL) {
        tensor_free(layer->output);
    }

    layer->output = Y;
    return layer->output;
}

Tensor *linear_layer_backward(LinearLayer *layer, const Tensor *upstream_grad) {

    // 1. Weights -> grad_weights = upstream_grad ⊗ input^T (outer product) - accumulate
    tensor_outer_product_accumulate(layer->grad_weights, upstream_grad, layer->input);

    // 2. Bias -> grad_biases = upstream_grad - accumulate
    tensor_add(layer->grad_biases, layer->grad_biases, upstream_grad);

    // 3. Input Gradient -> dL/dinput = W^T * upstream_grad (return to previous layer)
    Tensor *grad_input = tensor_create1d(layer->input_size);
    tensor_matvec_mul_transpose(grad_input, layer->weights, upstream_grad);

    return grad_input;
}
