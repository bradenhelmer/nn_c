/*
 * activation_layer.c - Activation layer implementations.
 *
 */
#include "activations/activations.h"
#include "layer.h"
#include <stdlib.h>

Layer *activation_layer_create(ActivationType activation_type) {
    ActivationLayer *layer = (ActivationLayer *)malloc(sizeof(ActivationLayer));

    layer->activation_type = activation_type;
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

Tensor *activation_layer_forward(ActivationLayer *layer, const Tensor *input) {
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

    switch (layer->activation_type) {
    case ACTIVATION_SIGMOID:
        tensor_sigmoid(layer->output, input);
        break;
    case ACTIVATION_RELU:
        tensor_relu(layer->output, input);
        break;
    case ACTIVATION_TANH:
        tensor_tanh_activation(layer->output, input);
        break;
    case ACTIVATION_LINEAR:
        tensor_linear(layer->output, input);
        break;
    }

    return layer->output;
}

Tensor *activation_layer_backward(ActivationLayer *layer, const Tensor *upstream_grad) {
    // Compute gradient: dL/dinput = dL/doutput ⊙ f'(output)
    // Note: derivatives expect the activated output (e.g., sigmoid derivative uses s*(1-s))
    Tensor *grad_input = tensor_clone(layer->output);
    switch (layer->activation_type) {
    case ACTIVATION_SIGMOID:
        tensor_sigmoid_derivative(grad_input, layer->output);
        break;
    case ACTIVATION_RELU:
        tensor_relu_derivative(grad_input, layer->output);
        break;
    case ACTIVATION_TANH:
        tensor_tanh_derivative(grad_input, layer->output);
        break;
    case ACTIVATION_LINEAR:
        tensor_linear_derivative(grad_input, layer->output);
        break;
    }
    tensor_elementwise_mul(grad_input, upstream_grad, grad_input);

    return grad_input;
}
