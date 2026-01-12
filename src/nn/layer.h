/*
 * layer.h
 *
 * Layer struct and function declarations
 *
 * IMPORTANT: API ENCAPSULATION NOTICE
 * ------------------------------------
 * This header currently exposes all layer implementation structs (LinearLayer,
 * Conv2DLayer, etc.) which breaks encapsulation and prevents future changes
 * to internal representations without breaking client code.
 *
 * RECOMMENDED FUTURE REFACTOR:
 * - Move layer-specific structs to layer_internal.h
 * - Make them opaque in this public header (forward declarations only)
 * - Provide accessor functions where field access is needed
 * - Migrate API to use Layer * uniformly instead of LinearLayer *, etc.
 *
 * For now: Treat struct fields as PRIVATE - do not access directly in client code.
 * Use provided functions instead. Direct field access may break in future versions.
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

#endif /* ifndef LAYER_H */
