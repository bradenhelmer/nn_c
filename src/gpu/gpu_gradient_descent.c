/*
 * gpu_gradient_descent.c
 *
 * GPU Gradient descent implementations.
 */
#include "gpu_gradient_descent.h"
#include "data/batch.h"
#include "gpu/gpu_nn.h"
#include "gpu/gpu_tensor.h"
#include "utils/timing.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

TrainingResult *train_nn_gpu_batch(GPUNeuralNet *gpu_nn, Dataset *train_data, Dataset *val_data,
                                   TrainingConfig *config) {

    TrainingResult *result = (TrainingResult *)malloc(sizeof(TrainingResult));
    result->loss_history = malloc(config->max_epochs * sizeof(float));
    result->accuracy_history = malloc(config->max_epochs * sizeof(float));
    result->epochs_completed = config->max_epochs;
    Timer epoch_timer;

    // 2. Batch iterator
    BatchIterator *batch_iter = batch_iterator_create(train_data, config->batch_size);

    // Persistent GPU buffers for reuse each batch.
    GPUTensor *d_input_batch = gpu_tensor_create(4, (int[]){config->batch_size, 1, 28, 28});
    GPUTensor *d_target_batch = gpu_tensor_create(2, (int[]){config->batch_size, 10});

    // Pinned host memory for faster transfers
    float *h_input_pinned, *h_target_pinned;
    cudaMallocHost((void **)&h_input_pinned, config->batch_size * 784 * sizeof(float));
    cudaMallocHost((void **)&h_target_pinned, config->batch_size * 10 * sizeof(float));

    for (int epoch = 0; epoch < config->max_epochs; epoch++) {
        timer_start(&epoch_timer);
        float epoch_loss = 0.f;
        int samples_seen = 0;

        batch_iterator_shuffle(batch_iter);
        Batch *batch;

        // 3. Batch loop
        while ((batch = batch_iterator_next(batch_iter)) != NULL) {

            // 1. Transfer batch to GPU
            int actual_batch_size = batch->size;

            // Copy to pinned memory first
            memcpy(h_input_pinned, batch->X->data, actual_batch_size * 784 * sizeof(float));
            memcpy(h_target_pinned, batch->Y->data, actual_batch_size * 10 * sizeof(float));

            // Transfer to GPU
            gpu_tensor_copy_from_host(d_input_batch, h_input_pinned, actual_batch_size * 784);
            gpu_tensor_copy_from_host(d_target_batch, h_target_pinned, actual_batch_size * 10);

            // Update shape for partial batches
            d_input_batch->shape[0] = actual_batch_size;
            d_target_batch->shape[0] = actual_batch_size;

            // 2. Forward pass
            gpu_nn_zero_gradients(gpu_nn);
            GPUTensor *prediction = gpu_nn_forward(gpu_nn, d_target_batch);

            // 3. Compute loss (fused with backward start)
            float batch_loss = gpu_nn_compute_loss(gpu_nn, prediction, d_target_batch);
            epoch_loss += batch_loss * actual_batch_size;

            // 4. Backward pass
            gpu_nn_backward(gpu_nn, d_target_batch);

            // 5. Update weights
            gpu_nn_scale_gradients(gpu_nn, 1.0f / actual_batch_size);
            gpu_nn_optimizer_step(gpu_nn);

            samples_seen += actual_batch_size;
        }
        timer_stop(&epoch_timer);

        // 6. Divide metrics by total samples and not batches.
        result->loss_history[epoch] = epoch_loss / samples_seen;

        if (val_data != NULL) {
            result->accuracy_history[epoch] =
                gpu_nn_evaluate_accuracy(gpu_nn, val_data, d_input_batch, h_input_pinned);
        }

        batch_iterator_reset(batch_iter);

        if (config->verbose) {
            printf("Epoch %d: loss=%.4f, accuracy=%.2f%%, time=%.3fs\n", epoch,
                   result->loss_history[epoch], result->accuracy_history[epoch] * 100,
                   epoch_timer.elapsed);
        }
    }

    batch_iterator_free(batch_iter);
    gpu_tensor_free(d_input_batch);
    gpu_tensor_free(d_target_batch);
    cudaFreeHost(h_input_pinned);
    cudaFreeHost(h_target_pinned);

    result->final_loss = result->loss_history[result->epochs_completed - 1];
    return result;
}
