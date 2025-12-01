/*
 * loss.c
 *
 * Loss function implementations.
 */
#include "loss.h"
#include <assert.h>
#include <math.h>

float mse_loss(float predicted, float target) {
    float error = predicted - target;
    return 0.5f * error * error;
}

float vector_mse(const Vector *prediction, const Vector *target) {
    assert(prediction->size == target->size);
    float diff, sum = 0.f;
    for (int i = 0; i < prediction->size; i++) {
        diff = prediction->data[i] - target->data[i];
        sum += diff * diff;
    }
    return sum / prediction->size;
}

void vector_mse_derivative(Vector *result, const Vector *prediction, const Vector *target) {
    assert(prediction->size == result->size);
    assert(target->size == result->size);
    for (int i = 0; i < result->size; i++) {
        result->data[i] = (prediction->data[i] - target->data[i]) / result->size;
    }
}

#define EPSILON 1e-7f

float vector_cross_entropy(const Vector *prediction, const Vector *target) {
    assert(prediction->size == target->size);
    float sum = 0.f;
    for (int i = 0; i <= prediction->size; i++) {
        sum -= target->data[i] * logf(prediction->data[i] + EPSILON);
    }
    return sum;
}

void vector_cross_entropy_derivative(Vector *result, const Vector *prediction,
                                     const Vector *target) {
    assert(prediction->size == result->size);
    assert(target->size == result->size);
    for (int i = 0; i <= prediction->size; i++) {
        result->data[i] = -target->data[i] / (prediction->data[i] + EPSILON);
    }
}

const VectorLossPair VECTOR_MSE_LOSS = {.loss = vector_mse,
                                        .loss_derivative = vector_mse_derivative};

const VectorLossPair VECTOR_CROSS_ENTROPY_LOSS = {
    .loss = vector_cross_entropy, .loss_derivative = vector_cross_entropy_derivative};
