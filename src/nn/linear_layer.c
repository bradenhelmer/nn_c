/*
 * linear_layer.c - Linear layer implementations.
 *
 */
#include "../utils/utils.h"
#include "layer.h"
#include <math.h>
#include <stdlib.h>

LinearLayer *linear_layer_create(int input_size, int output_size, VectorActivationPair activation) {
    LinearLayer *layer = (LinearLayer *)malloc(sizeof(LinearLayer));

    layer->input_size = input_size;
    layer->output_size = output_size;
    layer->activation = activation;

    layer->weights = matrix_create(output_size, input_size);
    layer->biases = vector_zeros(output_size);

    layer->z = vector_create(output_size);
    layer->a = vector_create(output_size);
    layer->input = vector_create(input_size);

    layer->dW = matrix_create(output_size, input_size);
    layer->db = vector_create(output_size);
    layer->downstream_gradient = vector_create(input_size);

    return layer;
}

void linear_layer_free(LinearLayer *layer) {
    matrix_free(layer->weights);
    vector_free(layer->biases);

    vector_free(layer->z);
    vector_free(layer->a);
    vector_free(layer->input);

    matrix_free(layer->dW);
    vector_free(layer->db);
    vector_free(layer->downstream_gradient);

    free(layer);
}

static void _layer_init(LinearLayer *layer, float standard) {
    float min = -standard, max = standard;
    for (int i = 0; i < layer->weights->rows; i++) {
        for (int j = 0; j < layer->weights->cols; j++) {
            matrix_set(layer->weights, i, j, rand_rangef(min, max));
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
void linear_layer_forward(LinearLayer *layer, const Vector *input) {
    // Cache input for weight gradient calculation.
    vector_copy(layer->input, input);

    // z = Wx + b
    matrix_vector_multiply(layer->z, layer->weights, input);
    vector_add(layer->z, layer->z, layer->biases);

    // a = f(z)
    layer->activation.forward(layer->a, layer->z);
}

void linear_layer_backward(LinearLayer *layer, const Vector *upstream_grad) {

    // 1. Pre-activation -> dL/dz = dL/da ⊙ f'(z)
    Vector *dz = vector_create(layer->output_size);
    layer->activation.derivative(dz, layer->a);
    vector_elementwise_multiply(dz, upstream_grad, dz);

    // 2. Weights -> dL/dW = dz ⊗ input^T (outer product) - accumulate
    Matrix *dW_sample = matrix_create(layer->output_size, layer->input_size);
    vector_outer_product(dW_sample, dz, layer->input);
    matrix_add(layer->dW, layer->dW, dW_sample);
    matrix_free(dW_sample);

    // 3. Bias -> dL/db = dz - accumulate
    vector_add(layer->db, layer->db, dz);

    // 4. Downstream Gradient -> dL/da_prev = W^T
    matrix_transpose_vector_multiply(layer->downstream_gradient, layer->weights, dz);
    vector_free(dz);
}
