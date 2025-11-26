/*
 * activeations.c - Activation functions implementations
 */
#include "activations.h"
#include <assert.h>
#include <math.h>

float sigmoid_scalar(float x) {
    return 1.f / (1.f + expf(-(x)));
}

float sigmoid_scalar_derivative(float s) {
    return s * (1.f - s);
}

float relu_scalar(float x) {
    return x <= 0.f ? 0.f : x;
}

float relu_scalar_derivative(float x) {
    return x < 0.f ? 0.f : 1.f;
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

float linear_scalar_derivative(float x) {
    return 1.0f;
}

void vector_sigmoid(Vector *result, const Vector *input) {
    assert(input->size == result->size);
    for (int i = 0; i < input->size; i++) {
        result->data[i] = sigmoid_scalar(input->data[i]);
    }
}

void vector_sigmoid_derivative(Vector *result, const Vector *sigmoid_output) {
    assert(sigmoid_output->size == result->size);
    for (int i = 0; i < sigmoid_output->size; i++) {
        result->data[i] = sigmoid_scalar_derivative(sigmoid_output->data[i]);
    }
}

void vector_relu(Vector *result, const Vector *input) {
    assert(input->size == result->size);
    for (int i = 0; i < input->size; i++) {
        result->data[i] = relu_scalar(input->data[i]);
    }
}

void vector_relu_derivative(Vector *result, const Vector *input) {
    assert(input->size == result->size);
    for (int i = 0; i < input->size; i++) {
        result->data[i] = relu_scalar_derivative(input->data[i]);
    }
}

void vector_tanh_activation(Vector *result, const Vector *input) {
    assert(result->size == input->size);
    for (int i = 0; i < input->size; i++) {
        result->data[i] = tanh_scalar(input->data[i]);
    }
}
void vector_tanh_derivative(Vector *result, const Vector *tanh_output) {
    assert(result->size == tanh_output->size);
    for (int i = 0; i < tanh_output->size; i++) {
        result->data[i] = tanh_scalar_derivative(tanh_output->data[i]);
    }
}

void vector_softmax(Vector *result, const Vector *input) {
    assert(result->size == input->size);
    float max_val = vector_max(input);
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

void matrix_sigmoid(Matrix *result, const Matrix *input) {
    assert(result->rows == input->rows);
    assert(result->cols == input->cols);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            matrix_set(result, i, j, sigmoid_scalar(matrix_get(input, i, j)));
        }
    }
}

void matrix_sigmoid_derivative(Matrix *result, const Matrix *sigmoid_output) {
    assert(result->rows == sigmoid_output->rows);
    assert(result->cols == sigmoid_output->cols);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            matrix_set(result, i, j, sigmoid_scalar_derivative(matrix_get(sigmoid_output, i, j)));
        }
    }
}

void matrix_relu(Matrix *result, const Matrix *input) {
    assert(result->rows == input->rows);
    assert(result->cols == input->cols);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            matrix_set(result, i, j, relu_scalar(matrix_get(input, i, j)));
        }
    }
}
void matrix_relu_derivative(Matrix *result, const Matrix *input) {
    assert(result->rows == input->rows);
    assert(result->cols == input->cols);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            matrix_set(result, i, j, relu_scalar_derivative(matrix_get(input, i, j)));
        }
    }
}

const ActivationPair SIGMOID_ACTIVATION = {
    .forward = sigmoid_scalar, .derivative = sigmoid_scalar_derivative, .name = "sigmoid"};

const ActivationPair RELU_ACTIVATION = {
    .forward = relu_scalar, .derivative = relu_scalar_derivative, .name = "relu"};

const ActivationPair TANH_ACTIVATION = {
    .forward = tanh_scalar, .derivative = tanh_scalar_derivative, .name = "tanh"};

const ActivationPair LINEAR_ACTIVATION = {
    .forward = linear_scalar, .derivative = linear_scalar_derivative, .name = "linear"};
