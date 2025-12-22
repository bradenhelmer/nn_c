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

float tensor_mse(const Tensor *prediction, const Tensor *target) {
    assert(prediction->size == target->size);
    float diff, sum = 0.f;
    for (int i = 0; i < prediction->size; i++) {
        diff = prediction->data[i] - target->data[i];
        sum += diff * diff;
    }
    return sum / prediction->size;
}

void tensor_mse_derivative(Tensor *result, const Tensor *prediction, const Tensor *target) {
    assert(prediction->size == result->size);
    assert(target->size == result->size);
    for (int i = 0; i < result->size; i++) {
        result->data[i] = (prediction->data[i] - target->data[i]) / result->size;
    }
}

const TensorLossPair TENSOR_MSE_LOSS = {.loss = tensor_mse,
                                        .loss_derivative = tensor_mse_derivative};

#define EPSILON 1e-7f

float tensor_cross_entropy(const Tensor *prediction, const Tensor *target) {
    assert(prediction->size == target->size);
    float sum = 0.f;
    for (int i = 0; i < prediction->size; i++) {
        sum -= target->data[i] * logf(prediction->data[i] + EPSILON);
    }
    return sum;
}

void tensor_cross_entropy_derivative(Tensor *result, const Tensor *prediction,
                                     const Tensor *target) {
    assert(prediction->size == result->size);
    assert(target->size == result->size);
    for (int i = 0; i < prediction->size; i++) {
        result->data[i] = -target->data[i] / (prediction->data[i] + EPSILON);
    }
}

const TensorLossPair TENSOR_CROSS_ENTROPY_LOSS = {
    .loss = tensor_cross_entropy, .loss_derivative = tensor_cross_entropy_derivative};

// Helper: compute softmax on a 1D tensor in-place
static void _tensor_softmax(Tensor *result, const Tensor *input) {
    assert(result->size == input->size);
    float max_val = tensor_max(input);
    float sum = 0.0f;
    for (int i = 0; i < input->size; i++) {
        result->data[i] = expf(input->data[i] - max_val);
        sum += result->data[i];
    }
    for (int i = 0; i < result->size; i++) {
        result->data[i] /= sum;
    }
}

// Softmax cross-entropy loss: applies softmax to logits, then cross-entropy
float tensor_softmax_cross_entropy(const Tensor *logits, const Tensor *target) {
    assert(logits->size == target->size);

    // Apply softmax to logits
    Tensor *softmax_output = tensor_clone(logits);
    _tensor_softmax(softmax_output, logits);

    // Compute cross-entropy: -sum(target * log(softmax))
    float loss = tensor_cross_entropy(softmax_output, target);

    tensor_free(softmax_output);
    return loss;
}

// Gradient of softmax cross-entropy w.r.t. logits: softmax(logits) - target
void tensor_softmax_cross_entropy_derivative(Tensor *result, const Tensor *logits,
                                             const Tensor *target) {
    assert(logits->size == result->size);
    assert(target->size == result->size);

    // Apply softmax to logits
    _tensor_softmax(result, logits);

    // Gradient: softmax(logits) - target
    for (int i = 0; i < result->size; i++) {
        result->data[i] -= target->data[i];
    }
}

const TensorLossPair TENSOR_SOFTMAX_CROSS_ENTROPY_LOSS = {
    .loss = tensor_softmax_cross_entropy,
    .loss_derivative = tensor_softmax_cross_entropy_derivative};
