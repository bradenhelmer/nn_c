/*
 * gpu_layer_ops.h - Declarations for GPU forward/backward layer operations.
 *
 */
#ifndef GPU_LAYER_OPS_H
#define GPU_LAYER_OPS_H
#include "activations/activations.h"
#include "gpu/gpu_tensor.h"
#include "layers/layer.h"
#include <cublas_v2.h>

Conv2DParams gpu_conv2d_params_create(const Conv2DLayer *layer, const GPUTensor *input);
Conv2DParams gpu_conv2d_params_from_padded(const Conv2DLayer *layer, const GPUTensor *padded_input);
Conv2DParams gpu_conv2d_params_make(const Conv2DLayer *layer, int batch_size, int H_in, int W_in);
Conv2DParams gpu_conv2d_params_from_upstream(const Conv2DLayer *layer,
                                             const GPUTensor *upstream_grad);
#ifdef __cplusplus
extern "C" {
#endif

// Linear Layer
GPUTensor *gpu_linear_layer_forward(cublasHandle_t cublas, GPUTensor *Y, const GPUTensor *X,
                                    const GPUTensor *W, const GPUTensor *b);
GPUTensor *gpu_linear_layer_backward(cublasHandle_t cublas, GPUTensor *dY, const GPUTensor *X,
                                     const GPUTensor *dX, const GPUTensor *W, const GPUTensor *dW,
                                     const GPUTensor *db);

// Conv2D Layer

GPUTensor *gpu_conv2d_layer_forward(cublasHandle_t cublas, Conv2DLayer *layer,
                                    GPUTensor *input_cache_ptr, GPUTensor *Y, const GPUTensor *X,
                                    const GPUTensor *W, const GPUTensor *b);
GPUTensor *gpu_conv2d_layer_backward(cublasHandle_t cublas, Conv2DLayer *layer, GPUTensor *dY,
                                     const GPUTensor *X_col, const GPUTensor *dX,
                                     const GPUTensor *W, const GPUTensor *dW, const GPUTensor *db);

// Maxpool Layer
GPUTensor *gpu_maxpool_layer_forward(MaxPoolLayer *layer, GPUTensor *Y, const GPUTensor *X,
                                     int *max_indices);
GPUTensor *gpu_maxpool_layer_backward(MaxPoolLayer *layer, GPUTensor *dY, GPUTensor *dX,
                                      int *max_indices);

// Flatten Layer
GPUTensor *gpu_flatten_layer_forward(FlattenLayer *layer, GPUTensor *Y, const GPUTensor *X);
GPUTensor *gpu_flatten_layer_backward(FlattenLayer *layer, GPUTensor *dY, GPUTensor *dX);

// Reshape Layer
GPUTensor *gpu_reshape_layer_forward(ReshapeLayer *layer, GPUTensor *Y, const GPUTensor *X);
GPUTensor *gpu_reshape_layer_backward(ReshapeLayer *layer, GPUTensor *dY, GPUTensor *dX);

// Activation Layer
GPUTensor *gpu_activation_layer_forward(GPUTensor *output, GPUTensor *input,
                                        ActivationType activation_type);
GPUTensor *gpu_activation_layer_backward(GPUTensor *output, GPUTensor *upstream_grad,
                                         ActivationType activation_type);

#ifdef __cplusplus
}
#endif

#endif /* ifndef GPU_LAYER_OPS_H */
