
/*
 * fc_layer.c
 *
 * Fully connected layer implementations.
 */
#include "fc_layer.h"
#include "utils/utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Lifecycle
FCLayer *fc_layer_create(int input_size, int output_size) {
    FCLayer *layer = (FCLayer *)malloc(sizeof(FCLayer));
    layer->input_size = input_size;
    layer->output_size = output_size;

    layer->weights = tensor_create(2, (int[]){output_size, input_size});
    layer->biases = tensor_create(1, (int[]){output_size});

    layer->input = NULL;

    layer->d_weights = tensor_create(2, (int[]){output_size, input_size});
    layer->d_biases = tensor_create(1, (int[]){output_size});
    return layer;
}

void fc_layer_free(FCLayer *layer) {
    tensor_free(layer->weights);
    tensor_free(layer->biases);

    if (layer->input != NULL) {
        tensor_free(layer->input);
    }

    tensor_free(layer->d_weights);
    tensor_free(layer->d_biases);
    free(layer);
}

void fc_layer_init_weights(FCLayer *layer) {
    float standard = sqrtf(2.f / (layer->input_size + layer->output_size));
    for (int i = 0; i < layer->weights->size; i++) {
        layer->weights->data[i] = rand_rangef(-standard, standard);
    }
}

Tensor *fc_layer_forward(FCLayer *layer, Tensor *input) {
    // x: input vector, shape [n]
    // returns: output vector, shape [m]
    Tensor *y = tensor_zeros(1, (int[]){layer->output_size});
    for (int i = 0; i < layer->output_size; i++) {
        float sum = layer->biases->data[i];
        for (int j = 0; j < layer->input_size; j++) {
            sum += tensor_get2d(layer->weights, i, j) * input->data[j];
        }
        y->data[i] = sum;
    }
    layer->input = tensor_clone(input);
    return y;
}

Tensor *fc_layer_backward(FCLayer *layer, Tensor *upstream_grad) {
    // δ: upstream gradient, shape [m]
    // returns: gradient w.r.t. input, shape [n]
    for (int i = 0; i < layer->output_size; i++) {
        layer->d_biases->data[i] += upstream_grad->data[i];
    }
    for (int i = 0; i < layer->output_size; i++) {
        for (int j = 0; j < layer->input_size; j++) {
            layer->d_weights->data[tensor_index2d(layer->d_weights, i, j)] +=
                upstream_grad->data[i] * layer->input->data[j];
        }
    }
    Tensor *dx = tensor_zeros(1, (int[]){layer->input_size});
    for (int j = 0; j < layer->input_size; j++) {
        float sum = 0.0f;
        for (int i = 0; i < layer->output_size; i++) {
            sum += tensor_get2d(layer->weights, i, j) * upstream_grad->data[i];
        }
        dx->data[j] = sum;
    }
    return dx;
}

// Optimizer will call this after weight update
void fc_layer_zero_gradients(FCLayer *layer) {
    memset(layer->d_weights->data, 0, sizeof(float) * layer->d_weights->size);
    memset(layer->d_biases->data, 0, sizeof(float) * layer->d_biases->size);
}
