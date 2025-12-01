/*
 * layer.h
 *
 * Single dense layer definitions.
 */
#ifndef LAYER_H
#define LAYER_H

#include "../activations/activations.h"
#include "../linalg/matrix.h"
#include "../linalg/vector.h"

typedef struct {
    int input_size;
    int output_size;
    Matrix *weights; // shape: (output_size, input_size)
    Vector *biases;  // shape: (output_size,)
    VectorActivationPair activation;

    // Cached for back propagation
    Vector *z;     // pre-activation: Wx + b
    Vector *a;     // post-activation: f(z)
    Vector *input; // cached input for weight gradients

    // Gradients
    Matrix *dW; // Weight gradients
    Vector *db; // Bias gradients
} Layer;

// Lifecycle
Layer *layer_create(int input_size, int output_size, VectorActivationPair activation);
void layer_free(Layer *layer);

// Forward/backward
Vector *layer_forward(Layer *layer, const Vector *input);
Vector *layer_backward(Layer *layer, const Vector *upstream_grad);

// Weight initialization
void layer_init_xavier(Layer *layer);
void layer_init_he(Layer *layer);

#endif /* ifndef LAYER_H */
