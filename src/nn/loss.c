/*
 * loss.c
 *
 * Loss function implementations.
 */
#include "loss.h"
#include "../activations/activations.h"
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

const VectorLossPair VECTOR_MSE_LOSS = {.loss = vector_mse,
                                        .loss_derivative = vector_mse_derivative};

#define EPSILON 1e-7f

float vector_cross_entropy(const Vector *prediction, const Vector *target) {
    assert(prediction->size == target->size);
    float sum = 0.f;
    for (int i = 0; i < prediction->size; i++) {
        sum -= target->data[i] * logf(prediction->data[i] + EPSILON);
    }
    return sum;
}

void vector_cross_entropy_derivative(Vector *result, const Vector *prediction,
                                     const Vector *target) {
    assert(prediction->size == result->size);
    assert(target->size == result->size);
    for (int i = 0; i < prediction->size; i++) {
        result->data[i] = -target->data[i] / (prediction->data[i] + EPSILON);
    }
}

const VectorLossPair VECTOR_CROSS_ENTROPY_LOSS = {
    .loss = vector_cross_entropy, .loss_derivative = vector_cross_entropy_derivative};

// Softmax cross-entropy loss: applies softmax to logits, then cross-entropy
float vector_softmax_cross_entropy(const Vector *logits, const Vector *target) {
    assert(logits->size == target->size);

    // Apply softmax to logits
    Vector *softmax_output = vector_create(logits->size);
    vector_softmax(softmax_output, logits);

    // Compute cross-entropy: -sum(target * log(softmax))
    float loss = vector_cross_entropy(softmax_output, target);

    vector_free(softmax_output);
    return loss;
}

// Gradient of softmax cross-entropy w.r.t. logits: softmax(logits) - target
void vector_softmax_cross_entropy_derivative(Vector *result, const Vector *logits,
                                             const Vector *target) {
    assert(logits->size == result->size);
    assert(target->size == result->size);

    // Apply softmax to logits
    vector_softmax(result, logits);

    // Gradient: softmax(logits) - target
    for (int i = 0; i < result->size; i++) {
        result->data[i] -= target->data[i];
    }
}

const VectorLossPair VECTOR_SOFTMAX_CROSS_ENTROPY_LOSS = {
    .loss = vector_softmax_cross_entropy,
    .loss_derivative = vector_softmax_cross_entropy_derivative};
