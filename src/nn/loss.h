/*
 * loss.h
 *
 * Loss function declarations.
 */

#ifndef LOSS_H
#define LOSS_H
#include "../tensor/tensor.h"

// Singular predictions
float mse_loss(float predicted, float target);

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

typedef float (*tensor_loss_function)(const Tensor *, const Tensor *);
typedef void (*tensor_loss_function_derivative)(Tensor *, const Tensor *, const Tensor *);

typedef struct {
    tensor_loss_function loss;
    tensor_loss_function_derivative loss_derivative;
} TensorLossPair;

extern const TensorLossPair TENSOR_MSE_LOSS;
extern const TensorLossPair TENSOR_CROSS_ENTROPY_LOSS;
extern const TensorLossPair TENSOR_SOFTMAX_CROSS_ENTROPY_LOSS;

#endif /* ifndef LOSS_H */
