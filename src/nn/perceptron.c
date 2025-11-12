/*
 * perceptron.c
 *
 * Perceptron function implementations.
 */
#include "perceptron.h"
#include <stdlib.h>

Perceptron *perceptron_create(int input_size, float min, float max, float learning_rate,
                              ActivationPair activation) {
    Perceptron *p = (Perceptron *)malloc(sizeof(Perceptron));
    p->weights = vector_random(input_size, min, max);
    p->bias = 0.f;
    p->learning_rate = learning_rate;
    p->input_size = input_size;
    p->activation = activation.forward;
    p->activation_prime = activation.derivative;
    p->last_output = 0.f;
    p->last_raw_output = 0.f;
    return p;
}

void perceptron_free(Perceptron *p) {
    if (p != NULL) {
        vector_free(p->weights);
        free(p);
    }
}

float perceptron_predict(Perceptron *p, const Vector *input) {
    float sum = vector_dot(p->weights, input) + p->bias;
    p->last_raw_output = sum;
    p->last_output = p->activation(sum);
    return p->last_output;
}

int perceptron_classify(Perceptron *p, const Vector *input, classifier_func classifier) {
    return classifier(perceptron_predict(p, input));
}
