/*
 * layer.h
 *
 * Layer struct and function declarations
 *
 * All layer types are opaque - see layer_internal.h for struct definitions.
 * Use the provided functions to interact with layers.
 */
#ifndef LAYER_H
#define LAYER_H

#include "activations/activations.h"
#include "tensor/tensor.h"

// =============================================================================
// LAYER GENERICS
// =============================================================================

typedef struct Layer Layer;

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
typedef struct LayerParameters LayerParameters;

// Returns a LayerParameters struct containing borrowed references to layer tensors
// Caller must call layer_parameters_free() to free the pairs array
// Do NOT free the individual param/grad_param tensors - the layer owns them
LayerParameters layer_get_parameters(Layer *layer);

// Frees the pairs array in LayerParameters
// Does NOT free the individual tensors (param/grad_param) - those belong to the layer
void layer_parameters_free(LayerParameters *params);

// =============================================================================
// LINEAR LAYER
// =============================================================================

typedef struct LinearLayer LinearLayer;

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

typedef struct ActivationLayer ActivationLayer;

// Lifecycle
Layer *activation_layer_create(ActivationType activation_type);
void activation_layer_free(ActivationLayer *layer);

// Forward/backward
Tensor *activation_layer_forward(ActivationLayer *layer, const Tensor *input);
Tensor *activation_layer_backward(ActivationLayer *layer, const Tensor *upstream_grad);

// =============================================================================
// CONVOLUTIONAL 2D LAYER
// =============================================================================

typedef struct Conv2DLayer Conv2DLayer;
typedef struct Conv2DParams Conv2DParams;

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

typedef struct MaxPoolLayer MaxPoolLayer;

// Lifecycle
Layer *maxpool_layer_create(int pool_size, int stride);
void maxpool_layer_free(MaxPoolLayer *layer);

Tensor *maxpool_layer_forward(MaxPoolLayer *layer, const Tensor *input);
Tensor *maxpool_layer_backward(MaxPoolLayer *layer, const Tensor *upstream_grad);

// =============================================================================
// FLATTEN LAYER
// =============================================================================

typedef struct FlattenLayer FlattenLayer;

Layer *flatten_layer_create();
void flatten_layer_free(FlattenLayer *layer);

Tensor *flatten_layer_forward(FlattenLayer *layer, const Tensor *input);
Tensor *flatten_layer_backward(FlattenLayer *layer, const Tensor *upstream_grad);

// =============================================================================
// RESHAPE LAYER
// =============================================================================

typedef struct ReshapeLayer ReshapeLayer;

Layer *reshape_layer_create(int target_ndim, const int *target_shape);
void reshape_layer_free(ReshapeLayer *layer);

Tensor *reshape_layer_forward(ReshapeLayer *layer, const Tensor *input);
Tensor *reshape_layer_backward(ReshapeLayer *layer, const Tensor *upstream_grad);

#endif /* ifndef LAYER_H */
