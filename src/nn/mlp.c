/*
 * mlp.c
 *
 * Multi-layer (perceptron) network implementations.
 */
#include "mlp.h"
#include "../data/dataset.h"
#include "layer.h"
#include "loss.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// Lifecycle
MLP *mlp_create(int num_layers, float learning_rate, TensorLossPair loss_pair,
                mlp_classifier classifier) {
    assert(num_layers > 1);
    MLP *mlp = (MLP *)malloc(sizeof(MLP));
    mlp->num_layers = num_layers;
    mlp->layers = (LinearLayer **)malloc(sizeof(LinearLayer *) * num_layers);
    mlp->learning_rate = learning_rate;
    mlp->loss = loss_pair;
    mlp->classifier = classifier;
    return mlp;
}

void mlp_add_layer(MLP *mlp, int index, LinearLayer *layer) {
    mlp->layers[index] = layer;
}

void mlp_free(MLP *mlp) {
    for (int i = 0; i < mlp->num_layers; i++) {
        linear_layer_free(mlp->layers[i]);
    }
    free(mlp->layers);
    free(mlp);
}

Tensor *mlp_forward(MLP *mlp, const Tensor *input) {

    const Tensor *current = input;

    // Chain remaining layer outputs.
    for (int i = 0; i < mlp->num_layers; i++) {
        LinearLayer *layer = mlp->layers[i];
        linear_layer_forward(layer, current);
        current = (const Tensor *)layer->a;
    }

    return (Tensor *)current;
}

void mlp_backward(MLP *mlp, const Tensor *target) {

    // 1. Compute initial gradient from loss
    LinearLayer *prev = mlp->layers[mlp->num_layers - 1];
    Tensor *output = prev->a;
    Tensor *gradient = tensor_clone(output);
    mlp->loss.loss_derivative(gradient, output, target);

    // 2. Run first layer backward.
    linear_layer_backward(prev, gradient);

    // 2. Propagate backward through layers
    for (int i = mlp->num_layers - 2; i >= 0; i--) {
        LinearLayer *layer = mlp->layers[i];
        linear_layer_backward(layer, prev->downstream_gradient);
        prev = layer;
    }

    tensor_free(gradient);
}

void mlp_zero_gradients(MLP *mlp) {
    for (int i = 0; i < mlp->num_layers; i++) {
        LinearLayer *layer = mlp->layers[i];
        tensor_fill(layer->dW, 0.f);
        tensor_fill(layer->db, 0.f);
        tensor_fill(layer->downstream_gradient, 0.f);
    }
}

void mlp_update_weights(MLP *mlp) {
    for (int i = 0; i < mlp->num_layers; i++) {
        LinearLayer *layer = mlp->layers[i];

        // Update weights: W = W - (lr * dW)
        for (int j = 0; j < layer->weights->size; ++j) {
            layer->weights->data[j] -= mlp->learning_rate * layer->dW->data[j];
        }

        // Update biases: b = b - (lr * db)
        for (int j = 0; j < layer->biases->size; ++j) {
            layer->biases->data[j] -= mlp->learning_rate * layer->db->data[j];
        }
    }
}

void mlp_scale_gradients(MLP *mlp, float scale) {
    for (int l = 0; l < mlp->num_layers; l++) {
        LinearLayer *layer = mlp->layers[l];
        tensor_scale(layer->dW, layer->dW, scale);
        tensor_scale(layer->db, layer->db, scale);
    }
}

void mlp_add_l2_gradient(MLP *mlp, float lambda) {
    for (int i = 0; i < mlp->num_layers; ++i) {
        LinearLayer *layer = mlp->layers[i];
        for (int j = 0; j < layer->weights->size; j++) {
            layer->dW->data[j] += lambda * layer->weights->data[j];
        }
    }
}

void test_mlp_on_dataset(MLP *mlp, Dataset *data, const char *name) {
    printf("\nTesting %s:\n\n", name);

    // Pre-allocate tensors for reuse
    int input_shape[] = {data->X->shape[1]};
    int output_shape[] = {data->Y->shape[1]};
    Tensor *input = tensor_zeros(1, input_shape);
    Tensor *target = tensor_zeros(1, output_shape);
    Tensor *classification = tensor_zeros(1, output_shape);

    for (int i = 0; i < data->num_samples; i++) {
        tensor_get_row(input, data->X, i);
        tensor_get_row(target, data->Y, i);

        Tensor *prediction = mlp_forward(mlp, input);
        mlp->classifier(classification, prediction);

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
