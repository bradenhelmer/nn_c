/*
 * layer_internal.h - Internal layer struct definitions.
 */

#ifndef LAYER_INTERNAL_H
#define LAYER_INTERNAL_H
#include "activations/activations.h"
#include "core/tensor.h"
#include "layer.h"
#include <stdlib.h>

// =============================================================================
// LAYER GENERICS
// =============================================================================

// Layer types.
typedef enum {
    LAYER_LINEAR,
    LAYER_ACTIVATION,
    LAYER_CONV_2D,
    LAYER_MAX_POOL,
    LAYER_FLATTEN,
    LAYER_RESHAPE
} LayerType;

// Core layer struct.
struct Layer {
    LayerType type;
    void *layer;
};

static Layer *layer_create(LayerType type, void *layer) {
    Layer *L = (Layer *)(malloc(sizeof(Layer)));
    L->type = type;
    L->layer = layer;
    return L;
}

// =============================================================================
// LAYER PARAMETER GENERICS
// =============================================================================

// A single trainable parameter and gradient pair.
// NOTE: ParameterPair does NOT own the tensors - they belong to the layer
typedef struct {
    Tensor *param;      // Borrowed reference - do not free
    Tensor *grad_param; // Borrowed reference - do not free
} ParameterPair;

// All trainable parameters in a Layer
// NOTE: LayerParameters OWNS the pairs array but NOT the individual tensors
struct LayerParameters {
    ParameterPair *pairs; // Owned array - freed by layer_parameters_free()
    int num_pairs;
};

// =============================================================================
// LINEAR LAYER
// =============================================================================

struct LinearLayer {
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
};

// =============================================================================
// ACTIVATION LAYER
// =============================================================================

struct ActivationLayer {
    ActivationType activation_type;

    // Cached for backward pass
    Tensor *input;  // input to activation (pre-activation values)
    Tensor *output; // output from activation
};

// =============================================================================
// CONVOLUTIONAL 2D LAYER
// =============================================================================

struct Conv2DLayer {
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
};

// Convolution dimension parameters (computed from ConvLayer + input)
struct Conv2DParams {
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
};

// =============================================================================
// MAX POOLING LAYER
// =============================================================================

struct MaxPoolLayer {
    int pool_size;
    int stride;

    // Cached during forward for backward pass.
    int input_c, input_h, input_w;
    int output_h, output_w;
    int *max_indices; // Flat array: [C * H_out * W_out]
    Tensor *output;
};

// =============================================================================
// FLATTEN LAYER
// =============================================================================

struct FlattenLayer {
    int input_ndims;
    int *input_shape;
    Tensor *output;
};

// =============================================================================
// RESHAPE LAYER
// =============================================================================

struct ReshapeLayer {
    int target_ndim;   // Number of dimensions in target shape
    int *target_shape; // Target shape array (owned by layer)

    // Cached for backward pass
    int input_ndim;   // Original input dimensions
    int *input_shape; // Original input shape (owned by layer)
    Tensor *output;   // Output tensor
};

#endif /* ifndef LAYER_INTERNAL_H */
