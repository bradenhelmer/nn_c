/*
 * tensor_activations.c - Tensor activation implementations.
 */
#include "activations.h"
#include <assert.h>
#include <math.h>

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
