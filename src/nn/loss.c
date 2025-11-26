/*
 * loss.c
 *
 * Loss function implementations.
 */
#include "loss.h"

float mse_loss(float predicted, float target) {
    float error = predicted - target;
    return 0.5f * error * error;
}
