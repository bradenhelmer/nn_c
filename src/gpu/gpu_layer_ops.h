/*
 * gpu_layer_ops.h - Declarations for GPU forward/backward layer operations.
 *
 */
#ifndef GPU_LAYER_OPS_H
#define GPU_LAYER_OPS_H
#include "activations/activations.h"
#include "gpu/gpu_tensor.h"
#include <cublas_v2.h>

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

// Maxpool Layer

// Flatten Layer

// Activation Layer
GPUTensor *gpu_activation_layer_forward(GPUTensor *output, GPUTensor *input,
                                        ActivationType activation_type);
GPUTensor *gpu_activation_layer_backward(GPUTensor *output, GPUTensor *upstream_grad,
                                         ActivationType activation_type);

// Loss functions
void gpu_softmax_cross_entropy_loss(float *d_losses, const GPUTensor *prediction,
                                    const GPUTensor *target, const int batch_size,
                                    const int num_classes);

#ifdef __cplusplus
}
#endif

#endif /* ifndef GPU_LAYER_OPS_H */
