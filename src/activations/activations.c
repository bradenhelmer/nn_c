/*
 * activeations.c - Activation functions implementations
 */
#include "activations.h"
#include <assert.h>
#include <math.h>

float sigmoid(float x) {
    return 1.f / (1.f + expf(-(x)));
}

float sigmoid_derivative(float s) {
    return s * (1.f - s);
}

float relu(float x) {
    return x <= 0.f ? 0.f : x;
}

float relu_derivative(float x) {
    return x < 0.f ? 0.f : 1.f;
}

float tanh_derivative(float t) {
    return 1.f - (t * t);
}

void vector_sigmoid(Vector *result, const Vector *input) {
    assert(input->size == result->size);
    for (int i = 0; i < input->size; i++) {
        result->data[i] = sigmoid(input->data[i]);
    }
}

void vector_sigmoid_derivative(Vector *result, const Vector *sigmoid_output) {
    assert(sigmoid_output->size == result->size);
    for (int i = 0; i < sigmoid_output->size; i++) {
        result->data[i] = sigmoid_derivative(sigmoid_output->data[i]);
    }
}

void vector_relu(Vector *result, const Vector *input) {
    assert(input->size == result->size);
    for (int i = 0; i < input->size; i++) {
        result->data[i] = relu(input->data[i]);
    }
}

void vector_relu_derivative(Vector *result, const Vector *input) {
    assert(input->size == result->size);
    for (int i = 0; i < input->size; i++) {
        result->data[i] = relu_derivative(input->data[i]);
    }
}

void vector_tanh_activation(Vector *result, const Vector *input) {
    assert(result->size == input->size);
    for (int i = 0; i < input->size; i++) {
        result->data[i] = tanhf(input->data[i]);
    }
}
void vector_tanh_derivative(Vector *result, const Vector *tanh_output) {
    assert(result->size == tanh_output->size);
    for (int i = 0; i < tanh_output->size; i++) {
        result->data[i] = tanh_derivative(tanh_output->data[i]);
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
            matrix_set(result, i, j, sigmoid(matrix_get(input, i, j)));
        }
    }
}

void matrix_sigmoid_derivative(Matrix *result, const Matrix *sigmoid_output) {
    assert(result->rows == sigmoid_output->rows);
    assert(result->cols == sigmoid_output->cols);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            matrix_set(result, i, j, sigmoid_derivative(matrix_get(sigmoid_output, i, j)));
        }
    }
}

void matrix_relu(Matrix *result, const Matrix *input) {
    assert(result->rows == input->rows);
    assert(result->cols == input->cols);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            matrix_set(result, i, j, relu(matrix_get(input, i, j)));
        }
    }
}
void matrix_relu_derivative(Matrix *result, const Matrix *input) {
    assert(result->rows == input->rows);
    assert(result->cols == input->cols);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            matrix_set(result, i, j, relu_derivative(matrix_get(input, i, j)));
        }
    }
}
