/*
 * tensor_activations.c - Tensor activation implementations.
 */
#include "activations.h"

void tensor_relu_forward(Tensor *output, const Tensor *input) {
    for (int i = 0; i < input->size; i++) {
        output->data[i] = input->data[i] > 0.0f ? input->data[i] : 0.0f;
    }
}

void tensor_relu_backward(Tensor *grad_input, const Tensor *grad_output,
                          const Tensor *cached_forward_output) {
    for (int i = 0; i < grad_output->size; i++) {
        grad_input->data[i] = cached_forward_output->data[i] > 0.0f ? grad_input->data[i] : 0.0f;
    }
}
