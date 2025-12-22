/*
 * activations.h - Activation functions for neural networks
 *
 * Provides common activation functions (sigmoid, ReLU, tanh, softmax)
 * and their derivatives for both scalars and tensors, used in forward
 * and backward propagation.
 */

#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include "../tensor/tensor.h"

// =============================================================================
// SCALAR ACTIVATION FUNCTIONS
// =============================================================================

float sigmoid_scalar(float x);
float sigmoid_scalar_derivative(float s);
float relu_scalar(float x);
float relu_scalar_derivative(float x);
float tanh_scalar(float x);
float tanh_scalar_derivative(float t);
float linear_scalar(float x);
float linear_scalar_derivative(float x);

// Scalar Activation Types
typedef float (*scalar_activation_func)(float);
typedef float (*scalar_activation_derivative_func)(float);

typedef struct {
    scalar_activation_func forward;
    scalar_activation_derivative_func derivative;
    const char *name;
} ScalarActivationPair;

extern const ScalarActivationPair SIGMOID_ACTIVATION;
extern const ScalarActivationPair RELU_ACTIVATION;
extern const ScalarActivationPair TANH_ACTIVATION;
extern const ScalarActivationPair LINEAR_ACTIVATION;

// =============================================================================
// TENSOR ACTIVATION FUNCTIONS (for LinearLayer)
// =============================================================================

// Sigmoid
void tensor_sigmoid(Tensor *result, const Tensor *input);
void tensor_sigmoid_derivative(Tensor *result, const Tensor *sigmoid_output);

// ReLU
void tensor_relu(Tensor *result, const Tensor *input);
void tensor_relu_derivative(Tensor *result, const Tensor *input);

// Tanh
void tensor_tanh_activation(Tensor *result, const Tensor *input);
void tensor_tanh_derivative(Tensor *result, const Tensor *tanh_output);

// Linear (identity)
void tensor_linear(Tensor *result, const Tensor *input);
void tensor_linear_derivative(Tensor *result, const Tensor *input);

// Softmax (operates on whole 1D tensor)
void tensor_softmax(Tensor *result, const Tensor *input);

// Tensor Activation Types
typedef void (*tensor_activation_func)(Tensor *, const Tensor *);
typedef void (*tensor_activation_derivative_func)(Tensor *, const Tensor *);

typedef struct {
    tensor_activation_func forward;
    tensor_activation_derivative_func derivative;
    const char *name;
} TensorActivationPair;

extern const TensorActivationPair TENSOR_SIGMOID_ACTIVATION;
extern const TensorActivationPair TENSOR_RELU_ACTIVATION;
extern const TensorActivationPair TENSOR_TANH_ACTIVATION;
extern const TensorActivationPair TENSOR_LINEAR_ACTIVATION;

// =============================================================================
// CONV LAYER ACTIVATION FUNCTIONS (3-argument form for caching)
// =============================================================================

void tensor_relu_forward(Tensor *output, const Tensor *input);
void tensor_relu_backward(Tensor *grad_input, const Tensor *grad_output,
                          const Tensor *cached_forward_output);

#endif /* ifndef ACTIVATIONS_H */
