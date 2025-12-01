/*
 * loss.h
 *
 * Loss function declarations.
 */

#ifndef LOSS_H
#define LOSS_H
#include "../linalg/vector.h"

// Singular predictions
float mse_loss(float predicted, float target);

// Vector functions.
float vector_mse(const Vector *prediction, const Vector *target);
void vector_mse_derivative(Vector *result, const Vector *prediction, const Vector *target);
float vector_cross_entropy(const Vector *prediction, const Vector *target);
void vector_cross_entropy_derivative(Vector *result, const Vector *prediction,
                                     const Vector *target);

typedef float (*vector_loss_function)(const Vector *, const Vector *);
typedef void (*vector_loss_function_derivative)(Vector *, const Vector *, const Vector *);

typedef struct {
    vector_loss_function loss;
    vector_loss_function_derivative loss_derivative;
} VectorLossPair;

extern const VectorLossPair VECTOR_MSE_LOSS;
extern const VectorLossPair VECTOR_CROSS_ENTROPY_LOSS;

#endif /* ifndef LOSS_H */
