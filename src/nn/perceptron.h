/*
 * perceptron.h
 *
 * Structure definition and function declarations for perceptron operations.
 */
#ifndef PERCEPTRON_H
#define PERCEPTRON_H
#include "../linalg/vector.h"

typedef struct {
    Vector *weights;
    float bias;
    float learning_rate;
    int input_size;
} Perceptron;

// Creation, deletion, and initialization.
Perceptron *perceptron_create(int input_size, float learning_rate);
void perceptron_free(Perceptron *p);
void perceptron_initialize_weights(Perceptron *p);

// Forward pass
float perceptron_predict(Perceptron *p, const Vector *input);
int perceptron_classify(Perceptron *p, const Vector *input);

// Training
void perceptron_train_step(Perceptron *p, const Vector *input, float target);
void perceptron_update_weights(Perceptron *p, const Vector *input, float error);

#endif /* ifndef PERCEPTRON_H */
