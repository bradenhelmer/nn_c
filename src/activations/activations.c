/*
 * activations.c - Activation functions implementations
 */
#include "activations.h"
#include <assert.h>
#include <math.h>

// =============================================================================
// SCALAR ACTIVATION FUNCTIONS
// =============================================================================

float sigmoid_scalar(float x) {
    return 1.f / (1.f + expf(-(x)));
}

float sigmoid_scalar_derivative(float s) {
    return s * (1.f - s);
}

float relu_scalar(float x) {
    return x <= 0.f ? 0.f : x;
}

float relu_scalar_derivative(float relu_output) {
    return relu_output > 0.f ? 1.f : 0.f;
}

float tanh_scalar(float x) {
    return tanhf(x);
}

float tanh_scalar_derivative(float t) {
    return 1.f - (t * t);
}

float linear_scalar(float x) {
    return x;
}

float linear_scalar_derivative(__attribute__((unused)) float x) {
    return 1.0f;
}

const ScalarActivationPair SIGMOID_ACTIVATION = {
    .forward = sigmoid_scalar, .derivative = sigmoid_scalar_derivative, .name = "sigmoid"};

const ScalarActivationPair RELU_ACTIVATION = {
    .forward = relu_scalar, .derivative = relu_scalar_derivative, .name = "relu"};

const ScalarActivationPair TANH_ACTIVATION = {
    .forward = tanh_scalar, .derivative = tanh_scalar_derivative, .name = "tanh"};

const ScalarActivationPair LINEAR_ACTIVATION = {
    .forward = linear_scalar, .derivative = linear_scalar_derivative, .name = "linear"};

// =============================================================================
// TENSOR ACTIVATION FUNCTIONS
// =============================================================================
