/*
 * mlp.h
 *
 * Multi-layer (perceptron) network implementations.
 */
#include "../data/dataset.h"
#include "mlp.h"
#include "layer.h"
#include "loss.h"
#include <stdio.h>
#include <stdlib.h>

// Lifecycle
MLP *mlp_create(int num_layers, float learning_rate, VectorLossPair loss_pair,
                mlp_classifier classifier) {
    MLP *mlp = (MLP *)malloc(sizeof(MLP));
    mlp->num_layers = num_layers;
    mlp->layers = (Layer **)malloc(sizeof(Layer *) * num_layers);
    mlp->learning_rate = learning_rate;
    mlp->loss = loss_pair;
    mlp->classifier = classifier;
    return mlp;
}

void mlp_add_layer(MLP *mlp, int index, Layer *layer) {
    mlp->layers[index] = layer;
}

void mlp_free(MLP *mlp) {
    for (int i = 0; i < mlp->num_layers; i++) {
        layer_free(mlp->layers[i]);
    }
    free(mlp->layers);
    free(mlp);
}

Vector *mlp_forward(MLP *mlp, const Vector *input) {

    // Copy input into return vector.
    Vector *current = vector_create(input->size);
    vector_copy(current, input);

    // Chain layer outputs
    for (int i = 0; i < mlp->num_layers; i++) {
        Layer *layer = mlp->layers[i];
        current = layer_forward(layer, current);
    }

    // Return final output
    return current;
}

void mlp_backward(MLP *mlp, const Vector *target) {

    // 1. Compute initial gradient from loss
    Vector *output = mlp->layers[mlp->num_layers - 1]->a;
    Vector *gradient = vector_create(output->size);
    mlp->loss.loss_derivative(gradient, output, target);

    // 2. Propagate backward through layers
    for (int i = mlp->num_layers - 1; i >= 0; i--) {
        Layer *layer = mlp->layers[i];
        gradient = layer_backward(layer, target);
    }

    vector_free(gradient);
}

void mlp_zero_gradients(MLP *mlp) {
    for (int i = 0; i < mlp->num_layers; i++) {
        Layer *layer = mlp->layers[i];
        matrix_fill(layer->dW, 0.f);
        vector_fill(layer->db, 0.f);
    }
}

void mlp_update_weights(MLP *mlp) {
    for (int i = 0; i < mlp->num_layers; i++) {
        Layer *layer = mlp->layers[i];

        // 1. Update weights -> W = W - (lr * dW)
        matrix_scale(layer->dW, layer->dW, mlp->learning_rate);
        matrix_subtract(layer->weights, layer->weights, layer->dW);

        // 2. Update biases -> b = b - (lr * db)
        vector_scale(layer->db, layer->db, mlp->learning_rate);
        vector_subtract(layer->biases, layer->biases, layer->db);
    }
}

void test_mlp_on_dataset(MLP *mlp, Dataset *data, const char *name) {
    printf("\nTesting %s:\n\n", name);

    for (int i = 0; i < data->num_samples; i++) {
        Vector *input = get_row_as_vector(data->X, i);
        Vector *target = get_row_as_vector(data->Y, i);

        Vector *prediction = mlp_forward(mlp, input);
        Vector *classification = mlp->classifier(prediction);

        printf("Input: ");
        vector_print(input);
        printf(" -> Target: ");
        vector_print(target);
        printf(" , Predicted: ");
        vector_print(classification);
        printf(" (raw: ");
        vector_print(prediction);
        printf(")\n");

        vector_free(input);
        vector_free(target);
        vector_free(prediction);
        vector_free(classification);
    }
}
