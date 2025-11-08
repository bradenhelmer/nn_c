/*
 * activeations.c - Activation functions implementations
 */
#include "activations.h"
#include <assert.h>
#include <math.h>

void sigmoid(Vector *result, const Vector *input) {
    assert(input->size == result->size);
    for (int i = 0; i < input->size; i++) {
        result->data[i] = 1 / (1 + expf(-(input->data[i])));
    }
}
//
// void sigmoid_derivative(Vector *result, const Vector *sigmoid_output) {
// }
//
// void relu(Vector *result, const Vector *input) {
// }
// void relu_derivative(Vector *result, const Vector *input) {
// }
//
// void tanh_activation(Vector *result, const Vector *input) {
// }
// void tanh_derivative(Vector *result, const Vector *tanh_output) {
// }
//
// void softmax(Vector *result, const Vector *input) {
// }
//
// void matrix_sigmoid(Matrix *result, const Matrix *input) {
// }
// void matrix_sigmoid_derivative(Matrix *result, const Matrix *sigmoid_output) {
// }
// void matrix_relu(Matrix *result, const Matrix *input) {
// }
// void matrix_relu_derivative(Matrix *result, const Matrix *input) {
// }
