/*
 * gpu_loss.h - GPU loss function declarations.
 */
#ifndef GPU_LOSS_H
#define GPU_LOSS_H

#include "gpu/gpu_tensor.h"
#include "nn/loss.h"

#ifdef __cplusplus
extern "C" {
#endif

// ==============================================================================
// SPECIFIC LOSS FUNCTIONS
// ==============================================================================

// Softmax Cross-Entropy Loss
// Forward: Computes average loss across batch
// Input: logits (batch_size × num_classes), target (batch_size × num_classes, one-hot)
// Returns: scalar loss value
float gpu_softmax_cross_entropy_loss(const GPUTensor *logits, const GPUTensor *target);

// Softmax Cross-Entropy Backward
// Computes gradient: grad = softmax(logits) - target
// Input: logits (batch_size × num_classes), target (batch_size × num_classes)
// Output: grad (batch_size × num_classes), gradient w.r.t. logits
void gpu_softmax_cross_entropy_backward(GPUTensor *grad, const GPUTensor *logits,
                                        const GPUTensor *target);

// TODO: Add other loss types as needed (MSE, Cross-Entropy without softmax, etc.)

#ifdef __cplusplus
}
#endif

#endif /* GPU_LOSS_H */
