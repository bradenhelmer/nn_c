/*
 * conv_layer.h
 *
 * Convolutional Layer Declarations
 */
#ifndef CONV_LAYER_H
#define CONV_LAYER_H
#include "../tensor/tensor.h"

typedef struct {
    Tensor *kernels; // [C_out, C_in, k_h, k_w]
    Tensor *biases;  // [C_out]

    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;

    // Cache for backward pass
    Tensor *input;  // [C_in, H_in, W_in]
    Tensor *output; // [C_out, H_out, W_out]

    // Gradients
    Tensor *d_kernels; // Same shape as kernels
    Tensor *d_biases;  // Same shape as biases
} ConvLayer;

// Lifecycle
ConvLayer *conv_layer_create(int in_channels, int out_channels, int kernel_size, int stride,
                             int padding);
void conv_layer_free(ConvLayer *layer);
void conv_layer_init_weights(ConvLayer *layer);

// Forward/backward
Tensor *conv_layer_forward(ConvLayer *layer, Tensor *input);
Tensor *conv_layer_backward(ConvLayer *layer, Tensor *upstream_grad);

#endif /* ifndef CONV_LAYER_H */
