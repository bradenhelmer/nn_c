/*
 * reshape_layer.c - Reshape layer implementations.
 *
 */
#include "layer_internal.h"
#include "tensor/tensor_internal.h"
#include <stdlib.h>
#include <string.h>

Layer *reshape_layer_create(int target_ndim, const int *target_shape) {
    ReshapeLayer *rl = (ReshapeLayer *)malloc(sizeof(ReshapeLayer));

    // Copy target shape
    rl->target_ndim = target_ndim;
    rl->target_shape = (int *)malloc(sizeof(int) * target_ndim);
    memcpy(rl->target_shape, target_shape, sizeof(int) * target_ndim);

    // Initialize input shape tracking (will be set on first forward pass)
    rl->input_ndim = 0;
    rl->input_shape = NULL;
    rl->output = NULL;

    return layer_create(LAYER_RESHAPE, (void *)rl);
}

void reshape_layer_free(ReshapeLayer *layer) {
    if (layer->target_shape != NULL) {
        free(layer->target_shape);
    }
    if (layer->input_shape != NULL) {
        free(layer->input_shape);
    }
    if (layer->output != NULL) {
        tensor_free(layer->output);
    }
    free(layer);
}

Tensor *reshape_layer_forward(ReshapeLayer *layer, const Tensor *input) {
    // Cache input shape on first forward pass for backward
    if (layer->input_shape == NULL) {
        layer->input_ndim = input->ndim;
        layer->input_shape = (int *)malloc(sizeof(int) * input->ndim);
        memcpy(layer->input_shape, input->shape, sizeof(int) * input->ndim);
    }

    // Free previous output before creating new one
    if (layer->output != NULL) {
        tensor_free(layer->output);
    }

    // Reshape to target shape
    layer->output = tensor_unflatten(input, layer->target_ndim, layer->target_shape);
    return layer->output;
}

Tensor *reshape_layer_backward(ReshapeLayer *layer, const Tensor *upstream_grad) {
    // Reshape gradient back to input shape
    // This is the inverse operation of forward pass
    Tensor *view = tensor_unflatten(upstream_grad, layer->input_ndim, layer->input_shape);

    // Clone the view to create an independent tensor
    // This is necessary because nn_backward frees gradients after each layer,
    // but a view shares data with upstream_grad which would be freed too early
    Tensor *grad = tensor_clone(view);
    tensor_free(view);

    return grad;
}
