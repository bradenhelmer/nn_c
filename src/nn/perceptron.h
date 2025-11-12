/*
 * perceptron.h
 *
 * Structure definition and function declarations for perceptron operations.
 */
#ifndef PERCEPTRON_H
#define PERCEPTRON_H
#include "../activations/activations.h"
#include "../linalg/vector.h"

typedef struct {
    Vector *weights;
    float bias;
    float learning_rate;
    int input_size;
    activation_func activation;
    activation_derivative_func activation_prime;
    float last_output;     // Last activation cache
    float last_raw_output; // Pre-activation cache (z)
} Perceptron;

// Creation, deletion, and initialization.
Perceptron *perceptron_create(int input_size, float min, float max, float learning_rate,
                              ActivationPair activation);
void perceptron_free(Perceptron *p);

// Forward pass
float perceptron_predict(Perceptron *p, const Vector *input);

// Perceptron classifer func ptr
typedef int (*classifier_func)(float);

int perceptron_classify(Perceptron *p, const Vector *input, classifier_func classifier);

// Training
void perceptron_train_step(Perceptron *p, const Vector *input, float target);
void perceptron_update_weights(Perceptron *p, const Vector *input, float error);

#endif /* ifndef PERCEPTRON_H */
