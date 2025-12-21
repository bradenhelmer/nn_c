/*
 * layer.h
 *
 * Layer struct and function declarations
 */
#ifndef LAYER_H
#define LAYER_H

#include "../activations/activations.h"
#include "../linalg/matrix.h"
#include "../linalg/vector.h"

// =============================================================================
// LAYER GENERICS
// =============================================================================

// Layer types.
typedef enum {
    LAYER_LINEAR,
    LAYER_CONV_2D,
    LAYER_MAX_POOL,
} LayerType;

// Core layer struct.
typedef struct {
    LayerType type;
    void *layer;
} Layer;

Layer *layer_create(LayerType type, void *layer);
void layer_free(Layer *layer);
void layer_forward(Layer *layer, Tensor *input);
void layer_backward(Layer *layer, Tensor *upstream_grad);
void layer_init_weights(Layer *layer);

// =============================================================================
// LINEAR LAYER
// =============================================================================

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
    Matrix *dW;                  // Weight gradients
    Vector *db;                  // Bias gradients
    Vector *downstream_gradient; // Gradient to pass downstream -> dL/da_prev
} LinearLayer;

// Lifecycle
LinearLayer *linear_layer_create(int input_size, int output_size, VectorActivationPair activation);
void linear_layer_free(LinearLayer *layer);

// Forward/backward
void linear_layer_forward(LinearLayer *layer, const Vector *input);
void linear_layer_backward(LinearLayer *layer, const Vector *upstream_grad);

// Weight initialization
void linear_layer_init_xavier(LinearLayer *layer);
void linear_layer_init_he(LinearLayer *layer);

// =============================================================================
// CONVOLUTIONAL 2D LAYER
// =============================================================================

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

// =============================================================================
// MAX POOLING LAYER
// =============================================================================

typedef struct {
    int pool_size;
    int stride;

    // Cached during forward for backward pass.
    int input_c, input_h, input_w;
    int output_h, output_w;
    int *max_indices; // Flat array: [C * H_out * W_out]
} MaxPoolLayer;

// Lifecycle
MaxPoolLayer *maxpool_create(int pool_size, int stride);
void maxpool_free(MaxPoolLayer *layer);

// Forward pass: returns output tensor, caches max_indices internally.
Tensor *maxpool_forward(MaxPoolLayer *layer, Tensor *input);

// Backward pass: uses cached max_indices to route gradients.
Tensor *maxpool_backward(MaxPoolLayer *layer, Tensor *upstream_grad);

// Index helpers

// 3D -> 1D index for output/max_indices
static inline int out_idx(int c, int i, int j, int H_out, int W_out) {
    return c * (H_out * W_out) + i * W_out + j;
}

// Encode (m, n) -> flat window index
static inline int encode_window_idx(int m, int n, int pool_size) {
    return m * pool_size + n;
}

// Decode flat window index -> (m, n)
static inline void decode_window_index(int flat_idx, int pool_size, int *m, int *n) {
    *m = flat_idx / pool_size;
    *n = flat_idx % pool_size;
}

// =============================================================================
// FULLY CONNECTED LAYER
// =============================================================================

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

// Optimizer will call this after weight update
void fc_layer_zero_gradients(FCLayer *layer);

#endif /* ifndef LAYER_H */
