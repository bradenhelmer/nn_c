/*
 * regularization.h
 *
 * Regularization declarations.
 */

#ifndef REGULARIZATION_H
#define REGULARIZATION_H
#include "mlp.h"

typedef struct {
    float l2_lambda;    // L2 regularization strength
    float dropout_rate; // Probability of dropping (0 = no dropout)
    _Bool training_mode;

} RegConfig;

// L2 Regularization
float l2_penalty(MLP *mlp, float lambda);
float l2_gradient_contrib(Layer *layer, float lambda); // Adds lW to dW

// Dropout -> applied per-layer during forward pass
Vector *dropout_forward(Vector *input, float drop_rate, Vector **mask);
Vector *dropout_backward(Vector *grad, Vector *mask, float drop_rate);

#endif /* ifndef REGULARIZATION_H */
