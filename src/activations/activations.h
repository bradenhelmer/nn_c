/*
 * activations.h - Activation functions for neural networks
 *
 * Provides common activation functions (sigmoid, ReLU, tanh, softmax)
 * and their derivatives for both vectors and matrices, used in forward
 * and backward propagation.
 */

#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H
#include "../linalg/matrix.h"
#include "../linalg/vector.h"

// Activation functions (element-wise)
void sigmoid(Vector *result, const Vector *input);
void sigmoid_derivative(Vector *result, const Vector *sigmoid_output);

void relu(Vector *result, const Vector *input);
void relu_derivative(Vector *result, const Vector *input);

void tanh_activation(Vector *result, const Vector *input);
void tanh_derivative(Vector *result, const Vector *tanh_output);

// Softmax (special - operates on whole vector)
void softmax(Vector *result, const Vector *input);

// Matrix versions (apply element-wise)
void matrix_sigmoid(Matrix *result, const Matrix *input);
void matrix_sigmoid_derivative(Matrix *result, const Matrix *sigmoid_output);
void matrix_relu(Matrix *result, const Matrix *input);
void matrix_relu_derivative(Matrix *result, const Matrix *input);

#endif /* ifndef ACTIVATIONS_H */
