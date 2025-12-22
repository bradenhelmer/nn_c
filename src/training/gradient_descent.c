/*
 * dataset.c
 *
 * Gradient descent function implementations.
 */
#include "gradient_descent.h"
#include "../data/batch.h"
#include "../nn/loss.h"
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
    int input_shape[] = {train_data->X->shape[1]};
    Tensor *input = tensor_zeros(1, input_shape);

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

TrainingResult *train_mlp(MLP *mlp, Dataset *train_data, __attribute__((unused)) Dataset *val_data,
                          TrainingConfig *config) {

    // 1. Result Tracking
    TrainingResult *result = (TrainingResult *)malloc(sizeof(TrainingResult));
    result->loss_history = malloc(config->max_epochs * sizeof(float));
    result->accuracy_history = malloc(config->max_epochs * sizeof(float));
    result->epochs_completed = config->max_epochs;

    float prev_loss = INFINITY;

    // Pre-allocate tensors for reuse
    int input_shape[] = {train_data->X->shape[1]};
    int output_shape[] = {train_data->Y->shape[1]};
    Tensor *input = tensor_zeros(1, input_shape);
    Tensor *target = tensor_zeros(1, output_shape);
    Tensor *classification = tensor_zeros(1, output_shape);

    for (int epoch = 0; epoch < config->max_epochs; epoch++) {
        float epoch_loss = 0.f;
        int correct = 0;

        for (int i = 0; i < train_data->num_samples; i++) {
            mlp_zero_gradients(mlp);

            tensor_get_row(input, train_data->X, i);
            tensor_get_row(target, train_data->Y, i);
            Tensor *prediction = mlp_forward(mlp, input);
            mlp->classifier(classification, prediction);

            epoch_loss += mlp->loss.loss(prediction, target);
            if (tensor_equals(classification, target)) {
                correct++;
            }

            mlp_backward(mlp, target);
            mlp_update_weights(mlp);
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

TrainingResult *train_mlp_batch(MLP *mlp, Dataset *train_data,
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
    int input_shape[] = {train_data->X->shape[1]};
    int output_shape[] = {train_data->Y->shape[1]};
    Tensor *input = tensor_zeros(1, input_shape);
    Tensor *target = tensor_zeros(1, output_shape);
    Tensor *classification = tensor_zeros(1, output_shape);

    for (int epoch = 0; epoch < config->max_epochs; epoch++) {
        float epoch_loss = 0.f;
        int correct = 0;
        int samples_seen = 0;

        batch_iterator_shuffle(batch_iter);
        Batch *batch;

        // 3. Batch loop
        while ((batch = batch_iterator_next(batch_iter)) != NULL) {
            mlp_zero_gradients(mlp);

            // 4. Accumulate over batch samples
            for (int i = 0; i < batch->size; i++) {
                tensor_get_row(input, batch->X, i);
                tensor_get_row(target, batch->Y, i);
                Tensor *prediction = mlp_forward(mlp, input);
                mlp->classifier(classification, prediction);

                epoch_loss += mlp->loss.loss(prediction, target);
                if (tensor_equals(classification, target)) {
                    correct++;
                }

                mlp_backward(mlp, target);
            }

            samples_seen += batch->size;

            // 5. Average gradients and update weights
            mlp_scale_gradients(mlp, 1.0f / batch->size);
            mlp_update_weights(mlp);
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

TrainingResult *train_mlp_batch_opt(MLP *mlp, Dataset *train_data,
                                    __attribute__((unused)) Dataset *val_data,
                                    TrainingConfig *config) {

    // 1. Result Tracking
    TrainingResult *result = (TrainingResult *)malloc(sizeof(TrainingResult));
    result->loss_history = malloc(config->max_epochs * sizeof(float));
    result->accuracy_history = malloc(config->max_epochs * sizeof(float));
    result->epochs_completed = config->max_epochs;

    // Pre-allocate tensor buffers
    int input_shape[] = {train_data->X->shape[1]};
    int output_shape[] = {train_data->Y->shape[1]};
    Tensor *input = tensor_zeros(1, input_shape);
    Tensor *target = tensor_zeros(1, output_shape);
    Tensor *classification = tensor_zeros(1, output_shape);

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
            mlp_zero_gradients(mlp);

            // 4. Accumulate over batch samples
            for (int i = 0; i < batch->size; i++) {
                tensor_get_row(input, batch->X, i);
                tensor_get_row(target, batch->Y, i);
                Tensor *prediction = mlp_forward(mlp, input);
                mlp->classifier(classification, prediction);

                epoch_loss += mlp->loss.loss(prediction, target);
                if (tensor_equals(classification, target)) {
                    correct++;
                }

                mlp_backward(mlp, target);
            }

            samples_seen += batch->size;

            // 5. Average gradients and update weights using optimizer
            mlp_scale_gradients(mlp, 1.0f / batch->size);

            // L2 Regularization
            if (config->l2_lambda > 0.0f) {
                mlp_add_l2_gradient(mlp, config->l2_lambda);
            }

            optimizer_step(config->optimizer, mlp);
            batch_free(batch);
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
    }

    tensor_free(input);
    tensor_free(target);
    tensor_free(classification);

    batch_iterator_free(batch_iter);
    result->final_loss = result->loss_history[result->epochs_completed - 1];
    return result;
}
