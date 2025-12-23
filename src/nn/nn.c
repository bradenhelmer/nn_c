/*
 * nn.c
 *
 * Neural network implementations.
 */
#include "nn.h"
#include "../data/dataset.h"
#include "layer.h"
#include "loss.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// Lifecycle
NeuralNet *nn_create(int num_layers, float learning_rate, TensorLossPair loss_pair,
                     Classifier classifier) {
    assert(num_layers > 1);
    NeuralNet *nn = (NeuralNet *)malloc(sizeof(NeuralNet));
    nn->num_layers = num_layers;
    nn->layers = (Layer **)malloc(sizeof(Layer *) * num_layers);
    nn->learning_rate = learning_rate;
    nn->loss = loss_pair;
    nn->classifier = classifier;
    return nn;
}

void nn_add_layer(NeuralNet *nn, int index, Layer *layer) {
    nn->layers[index] = layer;
}

void nn_free(NeuralNet *nn) {
    for (int i = 0; i < nn->num_layers; i++) {
        layer_free(nn->layers[i]);
    }
    free(nn->layers);
    free(nn);
}

Tensor *nn_forward(NeuralNet *nn, const Tensor *input) {

    const Tensor *current = input;

    // Chain layer outputs
    for (int i = 0; i < nn->num_layers; i++) {
        Layer *layer = nn->layers[i];
        layer_forward(layer, current);
        current = (const Tensor *)layer_get_output(layer);
    }

    return (Tensor *)current;
}

void nn_backward(NeuralNet *nn, const Tensor *target) {

    // 1. Compute initial gradient from loss
    Tensor *output = layer_get_output(nn->layers[nn->num_layers - 1]);
    Tensor *gradient = tensor_clone(output);
    nn->loss.loss_derivative(gradient, output, target);

    // 2. Propagate backward through layers
    Tensor *current_grad = gradient;
    for (int i = nn->num_layers - 1; i >= 0; i--) {
        Tensor *next_grad = layer_backward(nn->layers[i], current_grad);
        if (current_grad != gradient) {
            tensor_free(current_grad);
        }
        current_grad = next_grad;
    }

    // Free the final gradient and initial loss gradient
    if (current_grad != NULL) {
        tensor_free(current_grad);
    }
    tensor_free(gradient);
}

void nn_zero_gradients(NeuralNet *nn) {
    for (int i = 0; i < nn->num_layers; i++) {
        layer_zero_gradients(nn->layers[i]);
    }
}

void nn_update_weights(NeuralNet *nn) {
    for (int i = 0; i < nn->num_layers; i++) {
        layer_update_weights(nn->layers[i], nn->learning_rate);
    }
}

void nn_scale_gradients(NeuralNet *nn, float scale) {
    for (int l = 0; l < nn->num_layers; l++) {
        layer_scale_gradients(nn->layers[l], scale);
    }
}

void nn_add_l2_gradient(NeuralNet *nn, float lambda) {
    for (int i = 0; i < nn->num_layers; ++i) {
        layer_add_l2_gradient(nn->layers[i], lambda);
    }
}

void test_nn_on_dataset(NeuralNet *nn, Dataset *data, const char *name) {
    printf("\nTesting %s:\n\n", name);

    // Pre-allocate tensors for reuse
    Tensor *input = tensor_create1d(data->X->shape[1]);
    Tensor *target = tensor_create1d(data->Y->shape[1]);
    Tensor *classification = tensor_create1d(data->Y->shape[1]);

    for (int i = 0; i < data->num_samples; i++) {
        tensor_get_row(input, data->X, i);
        tensor_get_row(target, data->Y, i);

        Tensor *prediction = nn_forward(nn, input);
        nn->classifier(classification, prediction);

        printf("Input: ");
        tensor_print(input);
        printf(" -> Target: ");
        tensor_print(target);
        printf(", Predicted: ");
        tensor_print(classification);
        printf(" (raw: ");
        tensor_print(prediction);
        printf(")\n");
    }

    tensor_free(input);
    tensor_free(target);
    tensor_free(classification);
}
