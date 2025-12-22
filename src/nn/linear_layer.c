/*
 * linear_layer.c - Linear layer implementations.
 *
 */
#include "../utils/utils.h"
#include "layer.h"
#include <math.h>
#include <stdlib.h>

LinearLayer *linear_layer_create(int input_size, int output_size, TensorActivationPair activation) {
    LinearLayer *layer = (LinearLayer *)malloc(sizeof(LinearLayer));

    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activation = activation;

    layer->weights = tensor_create2d(output_size, input_size);
    layer->biases = tensor_create1d(output_size);

    layer->z = tensor_create1d(output_size);
    layer->a = tensor_create1d(output_size);
    layer->input = tensor_create1d(input_size);

    layer->dW = tensor_create2d(output_size, input_size);
    layer->db = tensor_create1d(output_size);
    layer->downstream_gradient = tensor_create1d(input_size);

    return layer;
}

void linear_layer_free(LinearLayer *layer) {
    tensor_free(layer->weights);
    tensor_free(layer->biases);

    tensor_free(layer->z);
    tensor_free(layer->a);
    tensor_free(layer->input);

    tensor_free(layer->dW);
    tensor_free(layer->db);
    tensor_free(layer->downstream_gradient);

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

    // z = Wx + b
    tensor_matvec_mul(layer->z, layer->weights, input);
    tensor_add(layer->z, layer->z, layer->biases);

    // a = f(z)
    layer->activation.forward(layer->a, layer->z);
}

void linear_layer_backward(LinearLayer *layer, const Tensor *upstream_grad) {

    // 1. Pre-activation -> dL/dz = dL/da ⊙ f'(z)
    Tensor *dz = tensor_clone(layer->a);
    layer->activation.derivative(dz, layer->a);
    tensor_elementwise_mul(dz, upstream_grad, dz);

    // 2. Weights -> dL/dW = dz ⊗ input^T (outer product) - accumulate
    Tensor *dW_sample = tensor_create2d(layer->output_size, layer->input_size);
    tensor_outer_product(dW_sample, dz, layer->input);
    tensor_add(layer->dW, layer->dW, dW_sample);
    tensor_free(dW_sample);

    // 3. Bias -> dL/db = dz - accumulate
    tensor_add(layer->db, layer->db, dz);

    // 4. Downstream Gradient -> dL/da_prev = W^T * dz
    tensor_matvec_mul_transpose(layer->downstream_gradient, layer->weights, dz);
    tensor_free(dz);
}
