/*
 * perceptron.h
 *
 * Structure definition and function declarations for perceptron operations.
 */
#ifndef PERCEPTRON_H
#define PERCEPTRON_H
#include "../activations/activations.h"
#include "../data/dataset.h"
#include "../linalg/vector.h"

typedef int (*classifier_func)(float);

typedef struct {
    Vector *weights;
    float bias;
    float learning_rate;
    int input_size;
    activation_func activation;
    activation_derivative_func activation_prime;
    classifier_func classifier;
    float last_output;     // Last activation cache
    float last_raw_output; // Pre-activation cache (z)
} Perceptron;

// Creation, deletion, and initialization.
Perceptron *perceptron_create(int input_size, float min, float max, float learning_rate,
                              ActivationPair activation, classifier_func classifier);
void perceptron_free(Perceptron *p);

// Forward pass
float perceptron_predict(Perceptron *p, const Vector *input);

// Training
void perceptron_train_step(Perceptron *p, const Vector *input, float target);
void perceptron_update_weights(Perceptron *p, const Vector *input, float error);

// Testing
void test_perceptron_on_dataset(Perceptron *p, Dataset *data, const char *name);

#endif /* ifndef PERCEPTRON_H */
