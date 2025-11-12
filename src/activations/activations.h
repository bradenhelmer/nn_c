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
float sigmoid_scalar(float x);
float sigmoid_scalar_derivative(float s);
float relu_scalar(float x);
float relu_scalar_derivative(float x);
float tanh_scalar(float x);
float tanh_scalar_derivative(float t);
float linear_scalar(float x);
float linear_scalar_derivative(float x);

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

// Activation Types
typedef float (*activation_func)(float);
typedef float (*activation_derivative_func)(float);

typedef struct {
    activation_func forward;
    activation_derivative_func derivative;
    const char *name;
} ActivationPair;

extern const ActivationPair SIGMOID_ACTIVATION;
extern const ActivationPair RELU_ACTIVATION;
extern const ActivationPair TANH_ACTIVATION;
extern const ActivationPair LINEAR_ACTIVATION;

#endif /* ifndef ACTIVATIONS_H */
