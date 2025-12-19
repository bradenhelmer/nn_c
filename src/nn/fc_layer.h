/*
 * fc_layer.h
 *
 * Fully connected layer declarations
 */
#ifndef FC_LAYER_H
#define FC_LAYER_H
#include "tensor/tensor.h"

typedef struct {
    int input_size;  // n
    int output_size; // m

    Tensor *weights; // [m, n] (out x in)
    Tensor *biases;  // [m]

    // Cached for backward
    Tensor *input; // [n]

    // Gradients (accumulated, used by optimizer)
    Tensor *d_weights; // [m, n]
    Tensor *d_biases;  // [m]
} FCLayer;

// Lifecycle
FCLayer *fc_layer_create(int input_size, int output_size);
void fc_layer_free(FCLayer *layer);
void fc_layer_init_weights(FCLayer *layer);

// Forward/backward
Tensor *fc_layer_forward(FCLayer *layer, Tensor *input);
Tensor *fc_layer_backward(FCLayer *layer, Tensor *upstream_grad);

// Optimizer will ccall this after weight update
void fc_layer_zero_gradients(FCLayer *layer);

#endif /* ifndef FC_LAYER_H */
