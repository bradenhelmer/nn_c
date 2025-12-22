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
    int shape[] = {input_size};
    p->weights = tensor_random(1, shape, min, max);
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
        tensor_free(p->weights);
        free(p);
    }
}

float perceptron_predict(Perceptron *p, const Tensor *input) {
    float sum = tensor_dot(p->weights, input) + p->bias; // z
    p->last_raw_output = sum;                            // z
    p->last_output = p->activation(sum);                 // activation(z)
    return p->last_output;
}

void perceptron_train_step(Perceptron *p, const Tensor *input, float target) {
    float prediction = perceptron_predict(p, input);
    float error = prediction - target; // (ŷ - y) for gradient
    perceptron_update_weights(p, input, error);
}

void perceptron_update_weights(Perceptron *p, const Tensor *input, float error) {
    float delta = error * p->activation_prime(p->last_output); // (ŷ - y) * activation_prime(z)
    for (int i = 0; i < p->weights->size; i++) {
        p->weights->data[i] -= p->learning_rate * delta * input->data[i];
    }
    p->bias -= p->learning_rate * delta;
}

void test_perceptron_on_dataset(Perceptron *p, Dataset *data, const char *name) {
    printf("\nTesting %s:\n\n", name);

    int input_shape[] = {data->X->shape[1]};
    Tensor *input = tensor_zeros(1, input_shape);

    for (int i = 0; i < data->num_samples; i++) {
        tensor_get_row(input, data->X, i);
        float target = tensor_get2d(data->Y, i, 0);

        float prediction = perceptron_predict(p, input);
        int classification = p->classifier(prediction);

        printf("Input: [%.0f, %.0f] -> Target: %.0f, Predicted: %d (raw: %.4f)\n", input->data[0],
               input->data[1], target, classification, prediction);
    }

    tensor_free(input);
}
