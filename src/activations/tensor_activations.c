/*
 * tensor_activations.c - Tensor activation implementations.
 */
#include "activations.h"
#include "tensor/tensor_internal.h"
#include "utils/utils.h"
#include <assert.h>

void tensor_sigmoid(Tensor *output, const Tensor *input) {
    assert(input->size == output->size);
    for (int i = 0; i < input->size; i++) {
        output->data[i] = sigmoid_scalar(input->data[i]);
    }
}

void tensor_sigmoid_derivative(Tensor *output, const Tensor *sigmoid_output) {
    assert(sigmoid_output->size == output->size);
    for (int i = 0; i < sigmoid_output->size; i++) {
        output->data[i] = sigmoid_scalar_derivative(sigmoid_output->data[i]);
    }
}

void tensor_relu(Tensor *output, const Tensor *input) {
    assert(input->size == output->size);
    for (int i = 0; i < input->size; i++) {
        output->data[i] = relu_scalar(input->data[i]);
    }
}

void tensor_relu_derivative(Tensor *output, const Tensor *relu_output) {
    assert(relu_output->size == output->size);
    for (int i = 0; i < relu_output->size; i++) {
        output->data[i] = relu_scalar_derivative(relu_output->data[i]);
    }
}

void tensor_tanh(Tensor *output, const Tensor *input) {
    assert(output->size == input->size);
    for (int i = 0; i < input->size; i++) {
        output->data[i] = tanh_scalar(input->data[i]);
    }
}

void tensor_tanh_derivative(Tensor *output, const Tensor *tanh_output) {
    assert(output->size == tanh_output->size);
    for (int i = 0; i < tanh_output->size; i++) {
        output->data[i] = tanh_scalar_derivative(tanh_output->data[i]);
    }
}

void tensor_linear(Tensor *output, const Tensor *input) {
    assert(output->size == input->size);
    for (int i = 0; i < input->size; i++) {
        output->data[i] = input->data[i];
    }
}

void tensor_linear_derivative(Tensor *output, UNUSED const Tensor *input) {
    for (int i = 0; i < output->size; i++) {
        output->data[i] = 1.0f;
    }
}
