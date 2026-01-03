/*
 * layer.h
 *
 * Layer struct and function declarations
 */
#ifndef LAYER_H
#define LAYER_H

#include "../activations/activations.h"
#include "../tensor/tensor.h"
#include "gpu/gpu_tensor.h"

// =============================================================================
// LAYER GENERICS
// =============================================================================

// Layer types.
typedef enum {
    LAYER_LINEAR,
    LAYER_ACTIVATION,
    LAYER_CONV_2D,
    LAYER_MAX_POOL,
    LAYER_FLATTEN
} LayerType;

// Core layer struct.
typedef struct {
    LayerType type;
    void *layer;
} Layer;

Layer *layer_create(LayerType type, void *layer);
void layer_free(Layer *layer);
Tensor *layer_forward(Layer *layer, const Tensor *input);
Tensor *layer_backward(Layer *layer, const Tensor *upstream_grad);
Tensor *layer_get_output(Layer *layer);

// Generic weight operations
void layer_zero_gradients(Layer *layer);
void layer_update_weights(Layer *layer, float learning_rate);
void layer_scale_gradients(Layer *layer, float scale);
void layer_add_l2_gradient(Layer *layer, float lambda);

// =============================================================================
// LAYER PARAMETER GENERICS
// =============================================================================

// A single trainable parameter and gradient pair.
typedef struct {
    Tensor *param;
    Tensor *grad_param;
} ParameterPair;

// All trainable parameters in a Layer
typedef struct {
    ParameterPair *pairs;
    int num_pairs;
} LayerParameters;

LayerParameters layer_get_parameters(Layer *layer);
void layer_parameters_free(LayerParameters *params);

// =============================================================================
// LINEAR LAYER
// =============================================================================

typedef struct {
    int input_size;
    int output_size;
    Tensor *weights; // shape: (output_size, input_size)
    Tensor *biases;  // shape: (output_size,)

    // Cached for back propagation
    Tensor *output; // output: Wx + b
    Tensor *input;  // cached input for weight gradients

    // Gradients
    Tensor *grad_weights; // Weight gradients
    Tensor *grad_biases;  // Bias gradients
} LinearLayer;

// Lifecycle
Layer *linear_layer_create(int input_size, int output_size);
void linear_layer_free(LinearLayer *layer);

// Forward/backward
Tensor *linear_layer_forward(LinearLayer *layer, const Tensor *input);
Tensor *linear_layer_backward(LinearLayer *layer, const Tensor *upstream_grad);

// Weight initialization
void linear_layer_init_xavier(LinearLayer *layer);
void linear_layer_init_he(LinearLayer *layer);

// =============================================================================
// ACTIVATION LAYER
// =============================================================================

typedef struct {
    ActivationType activation_type;

    // Cached for backward pass
    Tensor *input;  // input to activation (pre-activation values)
    Tensor *output; // output from activation
} ActivationLayer;

// Lifecycle
Layer *activation_layer_create(ActivationType activation_type);
void activation_layer_free(ActivationLayer *layer);

// Forward/backward
Tensor *activation_layer_forward(ActivationLayer *layer, const Tensor *input);
Tensor *activation_layer_backward(ActivationLayer *layer, const Tensor *upstream_grad);

// =============================================================================
// CONVOLUTIONAL 2D LAYER
// =============================================================================

typedef struct {
    Tensor *weights; // [C_out, C_in, k_h, k_w] (renamed from kernels)
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
    Tensor *grad_weights; // Same shape as weights
    Tensor *grad_biases;  // Same shape as biases
} Conv2DLayer;

// Convolution dimension parameters (computed from ConvLayer + input)
typedef struct {
    int C_in;     // Input channels
    int C_out;    // Output channels
    int H_in;     // Input height (unpadded)
    int W_in;     // Input width (unpadded)
    int H_padded; // Padded input height
    int W_padded; // Padded input width
    int H_out;    // Output height
    int W_out;    // Output width
    int K;        // Kernel size
    int stride;   // Stride
    int padding;  // Padding
} Conv2DParams;

Conv2DParams conv2d_params_create(const Conv2DLayer *layer, const Tensor *input);
Conv2DParams conv2d_params_from_padded(const Conv2DLayer *layer, const Tensor *padded_input);
Conv2DParams conv2d_params_make(const Conv2DLayer *layer, int H_in, int W_in);
Conv2DParams conv2d_params_from_upstream(const Conv2DLayer *layer, const Tensor *upstream_grad);

// Lifecycle
Layer *conv2d_layer_create(int in_channels, int out_channels, int kernel_size, int stride,
                         int padding);
void conv2d_layer_free(Conv2DLayer *layer);
void conv2d_layer_init_weights(Conv2DLayer *layer);

// Forward/backward
Tensor *conv2d_layer_forward(Conv2DLayer *layer, const Tensor *input);
Tensor *conv2d_layer_forward_stride_optimized(Conv2DLayer *layer, const Tensor *input);
Tensor *conv2d_layer_backward(Conv2DLayer *layer, const Tensor *upstream_grad);
Tensor *conv2d_layer_backward_stride_optimized(Conv2DLayer *layer, const Tensor *upstream_grad);

// Im2Col Optimization
Tensor *conv2d_im2col(Conv2DLayer *layer, Tensor *X_pad);
Tensor *conv2d_col2im(Tensor *dX_col, const Conv2DParams *p);
Tensor *conv_layer_forward_im2col(Conv2DLayer *layer, const Tensor *input);
Tensor *conv_layer_backward_im2col(Conv2DLayer *layer, const Tensor *upstream_grad);

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
    Tensor *output;
} MaxPoolLayer;

// Lifecycle
Layer *maxpool_layer_create(int pool_size, int stride);
void maxpool_layer_free(MaxPoolLayer *layer);

Tensor *maxpool_layer_forward(MaxPoolLayer *layer, const Tensor *input);
Tensor *maxpool_layer_backward(MaxPoolLayer *layer, const Tensor *upstream_grad);

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
// FLATTEN LAYER
// =============================================================================

typedef struct {
    int input_ndims;
    int *input_shape;
    Tensor *output;
} FlattenLayer;

Layer *flatten_layer_create();
void flatten_layer_free(FlattenLayer *layer);

Tensor *flatten_layer_forward(FlattenLayer *layer, const Tensor *input);
Tensor *flatten_layer_backward(FlattenLayer *layer, const Tensor *upstream_grad);

#endif /* ifndef LAYER_H */
