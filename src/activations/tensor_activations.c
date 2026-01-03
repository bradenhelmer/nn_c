/*
 * tensor_activations.c - Tensor activation implementations.
 */
#include "activations.h"
#include <assert.h>

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
