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

// Activation functions (singular)
float sigmoid(float x);
float sigmoid_derivative(float s);
float relu(float x);
float relu_derivative(float x);
float tanh_derivative(float t);

// Activation functions (element-wise)
void vector_sigmoid(Vector *result, const Vector *input);
void vector_sigmoid_derivative(Vector *result, const Vector *sigmoid_output);

void vector_relu(Vector *result, const Vector *input);
void vector_relu_derivative(Vector *result, const Vector *input);

void vector_tanh_activation(Vector *result, const Vector *input);
void vector_tanh_derivative(Vector *result, const Vector *tanh_output);

// Softmax (special - operates on whole vector)
void vector_softmax(Vector *result, const Vector *input);

// Matrix versions (apply element-wise)
void matrix_sigmoid(Matrix *result, const Matrix *input);
void matrix_sigmoid_derivative(Matrix *result, const Matrix *sigmoid_output);
void matrix_relu(Matrix *result, const Matrix *input);
void matrix_relu_derivative(Matrix *result, const Matrix *input);

#endif /* ifndef ACTIVATIONS_H */
