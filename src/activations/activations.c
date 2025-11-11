/*
 * activeations.c - Activation functions implementations
 */
#include "activations.h"
#include <assert.h>
#include <math.h>

void sigmoid(Vector *result, const Vector *input) {
    assert(input->size == result->size);
    for (int i = 0; i < input->size; i++) {
        result->data[i] = 1.f / (1.f + expf(-(input->data[i])));
    }
}

void sigmoid_derivative(Vector *result, const Vector *sigmoid_output) {
    assert(sigmoid_output->size == result->size);
    for (int i = 0; i < sigmoid_output->size; i++) {
        float sig_out_val = sigmoid_output->data[i];
        result->data[i] = sig_out_val * (1.f - sig_out_val);
    }
}

void relu(Vector *result, const Vector *input) {
    assert(input->size == result->size);
    for (int i = 0; i < input->size; i++) {
        float x = input->data[i];
        result->data[i] = x <= 0.f ? 0.f : x;
    }
}

void relu_derivative(Vector *result, const Vector *input) {
    assert(input->size == result->size);
    for (int i = 0; i < input->size; i++) {
        float x = input->data[i];
        result->data[i] = x < 0.f ? 0.f : 1.f;
    }
}

void tanh_activation(Vector *result, const Vector *input) {
    assert(result->size == input->size);
    for (int i = 0; i < input->size; i++) {
        result->data[i] = tanhf(input->data[i]);
    }
}
void tanh_derivative(Vector *result, const Vector *tanh_output) {
    assert(result->size == tanh_output->size);
    for (int i = 0; i < tanh_output->size; i++) {
        float tanh_out_val = tanh_output->data[i];
        result->data[i] = 1.f - (tanh_out_val * tanh_out_val);
    }
}

void softmax(Vector *result, const Vector *input) {
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
            matrix_set(result, i, j, 1.f / (1.f + expf(-matrix_get(input, i, j))));
        }
    }
}

void matrix_sigmoid_derivative(Matrix *result, const Matrix *sigmoid_output) {
    assert(result->rows == sigmoid_output->rows);
    assert(result->cols == sigmoid_output->cols);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            float sig_out_val = matrix_get(sigmoid_output, i, j);
            matrix_set(result, i, j, sig_out_val * (1.f - sig_out_val));
        }
    }
}

void matrix_relu(Matrix *result, const Matrix *input) {
    assert(result->rows == input->rows);
    assert(result->cols == input->cols);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            float x = matrix_get(input, i, j);
            matrix_set(result, i, j, x <= 0.f ? 0.f : x);
        }
    }
}
void matrix_relu_derivative(Matrix *result, const Matrix *input) {
    assert(result->rows == input->rows);
    assert(result->cols == input->cols);
    for (int i = 0; i < result->rows; i++) {
        for (int j = 0; j < result->cols; j++) {
            float x = matrix_get(input, i, j);
            matrix_set(result, i, j, x < 0.f ? 0.f : 1.f);
        }
    }
}
