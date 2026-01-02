/*
 * gpu_layer_ops.h - Declarations for GPU forward/backward layer passes.
 *
 */
#ifndef GPU_LAYER_OPS_H
#define GPU_LAYER_OPS_H
#include "gpu/gpu_tensor.h"
#include <cublas_v2.h>

// Linear Layer
GPUTensor *gpu_linear_layer_forward(cublasHandle_t cublas, GPUTensor *Y, const GPUTensor *X,
                                    const GPUTensor *W, const GPUTensor *b);
GPUTensor *gpu_linear_layer_backward(cublasHandle_t cublas, GPUTensor *dY, const GPUTensor *X,
                                     const GPUTensor *W, const GPUTensor *dW, const GPUTensor *db);

// Conv2D Layer

// Maxpool Layer

// Flatten Layer

// Activations Layer

#endif /* ifndef GPU_LAYER_OPS_H */
