/*
 * loss.h
 *
 * Loss function declarations.
 */

#ifndef LOSS_H
#define LOSS_H
#include "core/tensor.h"

// Singular predictions
float mse_loss(float predicted, float target);

// =============================================================================
// TENSOR LOSS FUNCTIONS
// =============================================================================

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { LOSS_MSE, LOSS_CROSS_ENTROPY, LOSS_SOFTMAX_CROSS_ENTROPY } LossType;

// Tensor functions (1D tensors)
float tensor_mse(const Tensor *prediction, const Tensor *target);
void tensor_mse_derivative(Tensor *result, const Tensor *prediction, const Tensor *target);

// Cross Entropy
float tensor_cross_entropy(const Tensor *prediction, const Tensor *target);
void tensor_cross_entropy_derivative(Tensor *result, const Tensor *prediction,
                                     const Tensor *target);

// Softmax cross entropy
float tensor_softmax_cross_entropy(const Tensor *logits, const Tensor *target);
void tensor_softmax_cross_entropy_derivative(Tensor *result, const Tensor *logits,
                                             const Tensor *target);

#ifdef __cplusplus
}
#endif

#endif /* ifndef LOSS_H */
