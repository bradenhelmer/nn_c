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

// Scalar Activation Types
typedef float (*scalar_activation_func)(float);
typedef float (*scalar_activation_derivative_func)(float);

// Scalar Pairs
typedef struct {
    scalar_activation_func forward;
    scalar_activation_derivative_func derivative;
    const char *name;
} ScalarActivationPair;

extern const ScalarActivationPair SIGMOID_ACTIVATION;
extern const ScalarActivationPair RELU_ACTIVATION;
extern const ScalarActivationPair TANH_ACTIVATION;
extern const ScalarActivationPair LINEAR_ACTIVATION;

// Vector Activation Types
typedef void (*vector_activation_func)(Vector *, const Vector *);
typedef void (*vector_activation_derivative_func)(Vector *, const Vector *);

// Vector Pairs
typedef struct {
    vector_activation_func forward;
    vector_activation_derivative_func derivative;
    const char *name;
} VectorActivationPair;

extern const VectorActivationPair VECTOR_SIGMOID_ACTIVATION;
extern const VectorActivationPair VECTOR_RELU_ACTIVATION;
extern const VectorActivationPair VECTOR_TANH_ACTIVATION;

#endif /* ifndef ACTIVATIONS_H */
