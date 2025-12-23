/*
 * activation_layer.c - Activation layer implementations.
 *
 */
#include "layer.h"
#include <stdlib.h>

Layer *activation_layer_create(TensorActivationPair activation) {
    ActivationLayer *layer = (ActivationLayer *)malloc(sizeof(ActivationLayer));

    layer->activation = activation;
    layer->input = NULL;
    layer->output = NULL;

    return layer_create(LAYER_ACTIVATION, (void *)layer);
}

void activation_layer_free(ActivationLayer *layer) {
    if (layer->input != NULL) {
        tensor_free(layer->input);
    }
    if (layer->output != NULL) {
        tensor_free(layer->output);
    }
    free(layer);
}

void activation_layer_forward(ActivationLayer *layer, const Tensor *input) {
    // Free old cached values if they exist
    if (layer->input != NULL) {
        tensor_free(layer->input);
    }
    if (layer->output != NULL) {
        tensor_free(layer->output);
    }

    // Cache input for backward pass (needed for derivative computation)
    layer->input = tensor_clone(input);

    // Apply activation: output = f(input)
    layer->output = tensor_clone(input);
    layer->activation.forward(layer->output, input);
}

Tensor *activation_layer_backward(ActivationLayer *layer, const Tensor *upstream_grad) {
    // Compute gradient: dL/dinput = dL/doutput ⊙ f'(output)
    // Note: derivatives expect the activated output (e.g., sigmoid derivative uses s*(1-s))
    Tensor *grad_input = tensor_clone(layer->output);
    layer->activation.derivative(grad_input, layer->output);
    tensor_elementwise_mul(grad_input, upstream_grad, grad_input);

    return grad_input;
}
