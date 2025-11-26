/*
 * dataset.c
 *
 * Gradient descent function implementations.
 */
#include "gradient_descent.h"
#include "../nn/loss.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void training_result_free(TrainingResult *result) {
    free(result->loss_history);
    free(result->accuracy_history);
    free(result);
}

TrainingResult *train_perceptron(Perceptron *p, Dataset *train_data, Dataset *val_data,
                                 TrainingConfig *config) {

    // 1. Result Tracking
    TrainingResult *result = (TrainingResult *)malloc(sizeof(TrainingResult));
    result->loss_history = malloc(config->max_epochs * sizeof(float));
    result->accuracy_history = malloc(config->max_epochs * sizeof(float));
    result->epochs_completed = config->max_epochs;

    float prev_loss = INFINITY;

    // 2. Main training loop
    for (int epoch = 0; epoch < config->max_epochs; epoch++) {
        float epoch_loss = 0.f;
        int correct = 0;

        // 3. Iterating through dataset
        for (int i = 0; i < train_data->num_samples; i++) {

            // Sample i
            Vector *input = get_row_as_vector(train_data->X, i);
            float target = train_data->y->data[i];

            // Get prediction before training for loss/accuracy tracking
            float prediction = perceptron_predict(p, input);
            epoch_loss += mse_loss(prediction, target);
            if (p->classifier(prediction) == target) {
                correct++;
            }

            // Train on sample
            perceptron_train_step(p, input, target);

            // Free input vector
            vector_free(input);
        }

        result->loss_history[epoch] = epoch_loss / train_data->num_samples;
        result->accuracy_history[epoch] = (float)correct / train_data->num_samples;

        if (fabsf(prev_loss - result->loss_history[epoch]) < config->tolerance) {
            result->epochs_completed = epoch + 1;
            break;
        }
        prev_loss = result->loss_history[epoch];

        if (config->verbose) {
            printf("Epoch %d: loss=%.4f, accuracy=%.2f%%\n", epoch, result->loss_history[epoch],
                   result->accuracy_history[epoch] * 100);
        }
    }

    result->final_loss = result->loss_history[result->epochs_completed - 1];
    return result;
}
