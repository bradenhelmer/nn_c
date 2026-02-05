/*
 * loss.c
 *
 * Loss function implementations.
 */
#include "loss.h"
#include "core/tensor_internal.h"
#include <assert.h>
#include <math.h>
#include <stdlib.h>

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

// Helper: compute softmax on a single row of length num_classes
static void _softmax_row(float *output, const float *input, int num_classes) {
    float max_val = input[0];
    for (int i = 1; i < num_classes; i++) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }
    float sum = 0.0f;
    for (int i = 0; i < num_classes; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }
    for (int i = 0; i < num_classes; i++) {
        output[i] /= sum;
    }
}

// Softmax cross-entropy loss for batched inputs
// logits: (batch_size, num_classes) or (num_classes,) for single sample
// target: same shape as logits (one-hot encoded)
// Returns: average loss over batch
float tensor_softmax_cross_entropy(const Tensor *logits, const Tensor *target) {
    assert(logits->size == target->size);

    int batch_size, num_classes;
    if (logits->ndim == 1) {
        batch_size = 1;
        num_classes = logits->shape[0];
    } else {
        assert(logits->ndim == 2);
        batch_size = logits->shape[0];
        num_classes = logits->shape[1];
    }

    float total_loss = 0.0f;
    float *softmax_row = (float *)malloc(num_classes * sizeof(float));

    for (int b = 0; b < batch_size; b++) {
        const float *logits_row = logits->data + b * num_classes;
        const float *target_row = target->data + b * num_classes;

        // Compute softmax for this sample
        _softmax_row(softmax_row, logits_row, num_classes);

        // Cross-entropy for this sample: -sum(target * log(softmax))
        for (int c = 0; c < num_classes; c++) {
            total_loss -= target_row[c] * logf(softmax_row[c] + EPSILON);
        }
    }

    free(softmax_row);
    return total_loss / batch_size;
}

// Gradient of softmax cross-entropy w.r.t. logits
// result: same shape as logits
// Gradient per sample: (softmax(logits) - target) / batch_size
void tensor_softmax_cross_entropy_derivative(Tensor *result, const Tensor *logits,
                                             const Tensor *target) {
    assert(logits->size == result->size);
    assert(target->size == result->size);

    int batch_size, num_classes;
    if (logits->ndim == 1) {
        batch_size = 1;
        num_classes = logits->shape[0];
    } else {
        assert(logits->ndim == 2);
        batch_size = logits->shape[0];
        num_classes = logits->shape[1];
    }

    for (int b = 0; b < batch_size; b++) {
        const float *logits_row = logits->data + b * num_classes;
        const float *target_row = target->data + b * num_classes;
        float *result_row = result->data + b * num_classes;

        // Compute softmax for this sample
        _softmax_row(result_row, logits_row, num_classes);

        // Gradient: (softmax - target) / batch_size
        for (int c = 0; c < num_classes; c++) {
            result_row[c] = (result_row[c] - target_row[c]) / batch_size;
        }
    }
}
