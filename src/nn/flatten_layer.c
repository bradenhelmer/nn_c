/*
 * flatten_layer.c - Flattening layer implementations.
 */
#include "layer.h"
#include <stdlib.h>
#include <string.h>

Layer *flatten_layer_create() {
    FlattenLayer *fl = (FlattenLayer *)malloc(sizeof(FlattenLayer));
    fl->input_shape = NULL;
    fl->input_ndims = 0;
    fl->output = NULL;
    return layer_create(LAYER_FLATTEN, (void *)fl);
}

void flatten_layer_free(FlattenLayer *layer) {
    if (layer->input_shape != NULL) {
        free(layer->input_shape);
    }
    if (layer->output != NULL) {
        tensor_free(layer->output);
    }
    free(layer);
}

Tensor *flatten_layer_forward(FlattenLayer *layer, const Tensor *input) {
    if (layer->input_shape == NULL) {
        layer->input_ndims = input->ndim;
        layer->input_shape = (int *)malloc(sizeof(int) * input->ndim);
        memcpy(layer->input_shape, input->shape, sizeof(int) * input->ndim);
    }

    // Free previous output before creating new one.
    if (layer->output != NULL) {
        tensor_free(layer->output);
    }

    layer->output = tensor_flatten(input);
    return layer->output;
}

Tensor *flatten_layer_backward(FlattenLayer *layer, const Tensor *upstream_grad) {
    // Create a view with the original shape
    Tensor *view = tensor_unflatten(upstream_grad, layer->input_ndims, layer->input_shape);

    // Clone the view to create an independent tensor
    // This is necessary because nn_backward frees gradients after each layer,
    // but a view shares data with upstream_grad which would be freed too early
    Tensor *grad = tensor_clone(view);
    tensor_free(view);

    return grad;
}
