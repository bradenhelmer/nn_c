/*
 * layer.h
 *
 * Single dense layer implementations.
 */
#include "layer.h"
#include <math.h>
#include <stdlib.h>

Layer *layer_create(int input_size, int output_size, VectorActivationPair activation) {
    Layer *layer = (Layer *)malloc(sizeof(Layer));

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

void layer_free(Layer *layer) {
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

static void _layer_init(Layer *layer, float standard) {
    float min = -standard, max = standard;
    for (int i = 0; i < layer->weights->rows; i++) {
        for (int j = 0; j < layer->weights->cols; j++) {
            float norm_rand = (float)rand() / (float)RAND_MAX;
            matrix_set(layer->weights, i, j, min + norm_rand * (standard - min));
        }
    }
}

void layer_init_xavier(Layer *layer) {
    _layer_init(layer, sqrtf(2.f / (layer->input_size + layer->output_size)));
}
void layer_init_he(Layer *layer) {
    _layer_init(layer, sqrtf(2.f / layer->input_size));
}

// Forward/backward
void layer_forward(Layer *layer, const Vector *input) {
    // Cache input for weight gradient calculation.
    vector_copy(layer->input, input);

    // z = Wx + b
    matrix_vector_multiply(layer->z, layer->weights, input);
    vector_add(layer->z, layer->z, layer->biases);

    // a = f(z)
    layer->activation.forward(layer->a, layer->z);
}

void layer_backward(Layer *layer, const Vector *upstream_grad) {

    // 1. Pre-activation -> dL/dz = dL/da ⊙ f'(z)
    Vector *dz = vector_create(layer->output_size);
    layer->activation.derivative(dz, layer->a);
    vector_elementwise_multiply(dz, upstream_grad, dz);

    // 2. Weights -> dL/dW = dz ⊗ input^T (outer product)
    vector_outer_product(layer->dW, dz, layer->input);

    // 3. Bias -> dL/db = dz
    vector_copy(layer->db, dz);

    // 4. Downstream Gradient -> dL/da_prev = W^T
    matrix_transpose_vector_multiply(layer->downstream_gradient, layer->weights, dz);
    vector_free(dz);
}
