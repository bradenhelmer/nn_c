/*
 * dataset.c
 *
 * Gradient descent function implementations.
 */
#include "gradient_descent.h"
#include "../data/batch.h"
#include "../nn/loss.h"
#include "config.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

void training_result_free(TrainingResult *result) {
    free(result->loss_history);
    free(result->accuracy_history);
    free(result);
}

TrainingResult *train_perceptron(Perceptron *p, Dataset *train_data,
                                 __attribute__((unused)) Dataset *val_data,
                                 TrainingConfig *config) {

    // 1. Result Tracking
    TrainingResult *result = (TrainingResult *)malloc(sizeof(TrainingResult));
    result->loss_history = malloc(config->max_epochs * sizeof(float));
    result->accuracy_history = malloc(config->max_epochs * sizeof(float));
    result->epochs_completed = config->max_epochs;

    float prev_loss = INFINITY;

    // Pre-allocate input tensor
    Tensor *input = tensor_create1d(train_data->X->shape[1]);

    // 2. Main training loop
    for (int epoch = 0; epoch < config->max_epochs; epoch++) {
        float epoch_loss = 0.f;
        int correct = 0;

        // 3. Iterating through dataset
        for (int i = 0; i < train_data->num_samples; i++) {

            // Sample i
            tensor_get_row(input, train_data->X, i);
            float target = tensor_get2d(train_data->Y, i, 0);

            // Get prediction before training for loss/accuracy tracking
            float prediction = perceptron_predict(p, input);
            epoch_loss += mse_loss(prediction, target);
            if (p->classifier(prediction) == target) {
                correct++;
            }

            // Train on sample
            perceptron_train_step(p, input, target);
        }

        // Metrics Collection
        result->loss_history[epoch] = epoch_loss / train_data->num_samples;
        result->accuracy_history[epoch] = (float)correct / train_data->num_samples;

        // Check for early stop
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

    tensor_free(input);
    result->final_loss = result->loss_history[result->epochs_completed - 1];
    return result;
}

TrainingResult *train_nn(NeuralNet *nn, Dataset *train_data,
                         __attribute__((unused)) Dataset *val_data, TrainingConfig *config) {

    // 1. Result Tracking
    TrainingResult *result = (TrainingResult *)malloc(sizeof(TrainingResult));
    result->loss_history = malloc(config->max_epochs * sizeof(float));
    result->accuracy_history = malloc(config->max_epochs * sizeof(float));
    result->epochs_completed = config->max_epochs;

    float prev_loss = INFINITY;

    // Pre-allocate tensors for reuse
    Tensor *input = tensor_create1d(train_data->X->shape[1]);
    Tensor *target = tensor_create1d(train_data->Y->shape[1]);
    Tensor *classification = tensor_create1d(train_data->Y->shape[1]);

    for (int epoch = 0; epoch < config->max_epochs; epoch++) {
        float epoch_loss = 0.f;
        int correct = 0;

        for (int i = 0; i < train_data->num_samples; i++) {
            nn_zero_gradients(nn);

            tensor_get_row(input, train_data->X, i);
            tensor_get_row(target, train_data->Y, i);
            Tensor *prediction = nn_forward(nn, input);
            nn->classifier(classification, prediction);

            epoch_loss += nn->loss.loss(prediction, target);
            if (tensor_equals(classification, target)) {
                correct++;
            }

            nn_backward(nn, target);
            nn_update_weights(nn);
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

    tensor_free(input);
    tensor_free(target);
    tensor_free(classification);

    result->final_loss = result->loss_history[result->epochs_completed - 1];
    return result;
}

TrainingResult *train_nn_batch(NeuralNet *nn, Dataset *train_data,
                               __attribute__((unused)) Dataset *val_data, TrainingConfig *config) {

    // 1. Result Tracking
    TrainingResult *result = (TrainingResult *)malloc(sizeof(TrainingResult));
    result->loss_history = malloc(config->max_epochs * sizeof(float));
    result->accuracy_history = malloc(config->max_epochs * sizeof(float));
    result->epochs_completed = config->max_epochs;

    float prev_loss = INFINITY;

    // 2. Batch iterator
    BatchIterator *batch_iter = batch_iterator_create(train_data, config->batch_size);

    // Pre-allocate tensors for reuse
    Tensor *input = tensor_create1d(train_data->X->shape[1]);
    Tensor *target = tensor_create1d(train_data->Y->shape[1]);
    Tensor *classification = tensor_create1d(train_data->Y->shape[1]);

    for (int epoch = 0; epoch < config->max_epochs; epoch++) {
        float epoch_loss = 0.f;
        int correct = 0;
        int samples_seen = 0;

        batch_iterator_shuffle(batch_iter);
        Batch *batch;

        // 3. Batch loop
        while ((batch = batch_iterator_next(batch_iter)) != NULL) {
            nn_zero_gradients(nn);

            // 4. Accumulate over batch samples
            for (int i = 0; i < batch->size; i++) {
                tensor_get_row(input, batch->X, i);
                tensor_get_row(target, batch->Y, i);
                Tensor *prediction = nn_forward(nn, input);
                nn->classifier(classification, prediction);

                epoch_loss += nn->loss.loss(prediction, target);
                if (tensor_equals(classification, target)) {
                    correct++;
                }

                nn_backward(nn, target);
            }

            samples_seen += batch->size;

            // 5. Average gradients and update weights
            nn_scale_gradients(nn, 1.0f / batch->size);
            nn_update_weights(nn);
            batch_free(batch);
        }

        // 6. Divide metrics by total samples and not batches.
        result->loss_history[epoch] = epoch_loss / samples_seen;
        result->accuracy_history[epoch] = (float)correct / samples_seen;

        // 7. Early stop same as before.
        if (fabsf(prev_loss - result->loss_history[epoch]) < config->tolerance) {
            result->epochs_completed = epoch + 1;
            break;
        }

        batch_iterator_reset(batch_iter);

        prev_loss = result->loss_history[epoch];

        if (config->verbose) {
            printf("Epoch %d: loss=%.4f, accuracy=%.2f%%\n", epoch, result->loss_history[epoch],
                   result->accuracy_history[epoch] * 100);
        }
    }

    tensor_free(input);
    tensor_free(target);
    tensor_free(classification);

    batch_iterator_free(batch_iter);
    result->final_loss = result->loss_history[result->epochs_completed - 1];
    return result;
}

TrainingResult *train_nn_batch_opt(NeuralNet *nn, Dataset *train_data,
                                   __attribute__((unused)) Dataset *val_data,
                                   TrainingConfig *config) {

    // 1. Result Tracking
    TrainingResult *result = (TrainingResult *)malloc(sizeof(TrainingResult));
    result->loss_history = malloc(config->max_epochs * sizeof(float));
    result->accuracy_history = malloc(config->max_epochs * sizeof(float));
    result->epochs_completed = config->max_epochs;

    // Pre-allocate tensor buffers
    Tensor *input = tensor_create1d(train_data->X->shape[1]);
    Tensor *target = tensor_create1d(train_data->Y->shape[1]);
    Tensor *classification = tensor_create1d(train_data->Y->shape[1]);

    // 2. Batch iterator
    BatchIterator *batch_iter = batch_iterator_create(train_data, config->batch_size);

    for (int epoch = 0; epoch < config->max_epochs; epoch++) {
        float epoch_loss = 0.f;
        int correct = 0;
        int samples_seen = 0;

        batch_iterator_shuffle(batch_iter);
        Batch *batch;

        // 3. Batch loop
        while ((batch = batch_iterator_next(batch_iter)) != NULL) {
            nn_zero_gradients(nn);

            // 4. Accumulate over batch samples
            for (int i = 0; i < batch->size; i++) {
                tensor_get_row(input, batch->X, i);
                tensor_get_row(target, batch->Y, i);

                // Reshape for CNN (784,) -> (1, 28, 28)
                Tensor *spatial_input = tensor_unflatten(input, 3, (int[]){1, 28, 28});
                Tensor *prediction = nn_forward(nn, spatial_input);
                nn->classifier(classification, prediction);

                epoch_loss += nn->loss.loss(prediction, target);
                if (tensor_equals(classification, target)) {
                    correct++;
                }

                nn_backward(nn, target);
                tensor_free(spatial_input);
            }

            samples_seen += batch->size;

            // 5. Average gradients and update weights using optimizer
            nn_scale_gradients(nn, 1.0f / batch->size);

            // L2 Regularization
            if (config->l2_lambda > 0.0f) {
                nn_add_l2_gradient(nn, config->l2_lambda);
            }

            optimizer_step(config->optimizer, nn);
            batch_free(batch);
#if PROFILING
            if (samples_seen >= 1000) {
                break;
            }
#endif
        }

        // 6. Divide metrics by total samples and not batches.
        result->loss_history[epoch] = epoch_loss / samples_seen;
        result->accuracy_history[epoch] = (float)correct / samples_seen;

        batch_iterator_reset(batch_iter);

        // Update learning rate if scheduler present.
        if (config->scheduler != NULL) {
            scheduler_step(config->scheduler);
            optimizer_set_lr(config->optimizer, scheduler_get_lr(config->scheduler));
        }

        if (config->verbose) {
            printf("Epoch %d: loss=%.4f, accuracy=%.2f%%\n", epoch, result->loss_history[epoch],
                   result->accuracy_history[epoch] * 100);
        }
#if PROFILING
        if (samples_seen >= 1000) {
            break;
        }
#endif
    }

    tensor_free(input);
    tensor_free(target);
    tensor_free(classification);

    batch_iterator_free(batch_iter);
    result->final_loss = result->loss_history[result->epochs_completed - 1];
    return result;
}
