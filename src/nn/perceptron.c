/*
 * perceptron.c
 *
 * Perceptron function implementations.
 */
#include "perceptron.h"
#include <stdio.h>
#include <stdlib.h>

Perceptron *perceptron_create(int input_size, float min, float max, float learning_rate,
                              ScalarActivationPair activation, classifier_func classifier) {
    Perceptron *p = (Perceptron *)malloc(sizeof(Perceptron));
    p->weights = vector_random(input_size, min, max);
    p->bias = 0.f;
    p->learning_rate = learning_rate;
    p->input_size = input_size;
    p->activation = activation.forward;
    p->activation_prime = activation.derivative;
    p->classifier = classifier;
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
    float sum = vector_dot(p->weights, input) + p->bias; // z
    p->last_raw_output = sum;                            // z
    p->last_output = p->activation(sum);                 // activation(z)
    return p->last_output;
}

void perceptron_train_step(Perceptron *p, const Vector *input, float target) {
    float prediction = perceptron_predict(p, input);
    float error = prediction - target; // (ŷ - y) for gradient
    perceptron_update_weights(p, input, error);
}

void perceptron_update_weights(Perceptron *p, const Vector *input, float error) {
    float delta = error * p->activation_prime(p->last_output); // (ŷ - y) * activation_prime(z)
    for (int i = 0; i < p->weights->size; i++) {
        p->weights->data[i] -= p->learning_rate * delta * input->data[i];
    }
    p->bias -= p->learning_rate * delta;
}

void test_perceptron_on_dataset(Perceptron *p, Dataset *data, const char *name) {
    printf("\nTesting %s:\n\n", name);

    for (int i = 0; i < data->num_samples; i++) {
        Vector *input = get_row_as_vector(data->X, i);
        float target = data->Y->data[i];

        float prediction = perceptron_predict(p, input);
        int classification = p->classifier(prediction);

        printf("Input: [%.0f, %.0f] -> Target: %.0f, Predicted: %d (raw: %.4f)\n", input->data[0],
               input->data[1], target, classification, prediction);
        vector_free(input);
    }
}
