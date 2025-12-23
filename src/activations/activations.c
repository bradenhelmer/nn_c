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
    // ReLU output > 0 means input was > 0, so derivative is 1
    // ReLU output == 0 means input was <= 0, so derivative is 0
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

void tensor_sigmoid(Tensor *result, const Tensor *input) {
    assert(input->size == result->size);
    for (int i = 0; i < input->size; i++) {
        result->data[i] = sigmoid_scalar(input->data[i]);
    }
}

void tensor_sigmoid_derivative(Tensor *result, const Tensor *sigmoid_output) {
    assert(sigmoid_output->size == result->size);
    for (int i = 0; i < sigmoid_output->size; i++) {
        result->data[i] = sigmoid_scalar_derivative(sigmoid_output->data[i]);
    }
}

void tensor_relu(Tensor *result, const Tensor *input) {
    assert(input->size == result->size);
    for (int i = 0; i < input->size; i++) {
        result->data[i] = relu_scalar(input->data[i]);
    }
}

void tensor_relu_derivative(Tensor *result, const Tensor *relu_output) {
    assert(relu_output->size == result->size);
    for (int i = 0; i < relu_output->size; i++) {
        result->data[i] = relu_scalar_derivative(relu_output->data[i]);
    }
}

void tensor_tanh_activation(Tensor *result, const Tensor *input) {
    assert(result->size == input->size);
    for (int i = 0; i < input->size; i++) {
        result->data[i] = tanh_scalar(input->data[i]);
    }
}

void tensor_tanh_derivative(Tensor *result, const Tensor *tanh_output) {
    assert(result->size == tanh_output->size);
    for (int i = 0; i < tanh_output->size; i++) {
        result->data[i] = tanh_scalar_derivative(tanh_output->data[i]);
    }
}

void tensor_linear(Tensor *result, const Tensor *input) {
    assert(result->size == input->size);
    for (int i = 0; i < input->size; i++) {
        result->data[i] = input->data[i];
    }
}

void tensor_linear_derivative(Tensor *result, __attribute__((unused)) const Tensor *input) {
    for (int i = 0; i < result->size; i++) {
        result->data[i] = 1.0f;
    }
}

void tensor_softmax(Tensor *result, const Tensor *input) {
    assert(result->size == input->size);
    float max_val = tensor_max(input);
    float sum = 0.f;
    for (int i = 0; i < input->size; i++) {
        float exp_val = expf(input->data[i] - max_val);
        result->data[i] = exp_val;
        sum += exp_val;
    }
    for (int i = 0; i < input->size; i++) {
        result->data[i] /= sum;
    }
}

const TensorActivationPair TENSOR_SIGMOID_ACTIVATION = {
    .forward = tensor_sigmoid, .derivative = tensor_sigmoid_derivative, .name = "tensor_sigmoid"};

const TensorActivationPair TENSOR_RELU_ACTIVATION = {
    .forward = tensor_relu, .derivative = tensor_relu_derivative, .name = "tensor_relu"};

const TensorActivationPair TENSOR_TANH_ACTIVATION = {
    .forward = tensor_tanh_activation, .derivative = tensor_tanh_derivative, .name = "tensor_tanh"};

const TensorActivationPair TENSOR_LINEAR_ACTIVATION = {
    .forward = tensor_linear, .derivative = tensor_linear_derivative, .name = "tensor_linear"};
