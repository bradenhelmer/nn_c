/*
 * loss.h
 *
 * Loss function declarations.
 */

#ifndef LOSS_H
#define LOSS_H

// Singular predictions
float mse_loss(float predicted, float target);
float binary_cross_entropy(float predicted, float target);

// Batches
float compute_average_loss(float *predictions, float *targets, int size);

#endif /* ifndef LOSS_H */
