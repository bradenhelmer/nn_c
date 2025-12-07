/*
 * batch.c
 *
 * Batch training implementations.
 */
#include "batch.h"
#include "../utils/utils.h"
#include <math.h>
#include <stdlib.h>

void batch_free(Batch *b) {
    matrix_free(b->X);
    matrix_free(b->Y);
    free(b);
}

BatchIterator *batch_iterator_create(Dataset *data, int batch_size) {
    BatchIterator *batch_iter = (BatchIterator *)malloc(sizeof(BatchIterator));
    batch_iter->dataset = data;
    batch_iter->batch_size = batch_size;
    batch_iter->num_batches = ceil((double)(data->num_samples) / (double)(batch_size));
    batch_iter->current_idx = 0;
    batch_iter->indices = (int *)malloc(sizeof(int) * data->num_samples);
    for (int i = 0; i < data->num_samples; i++) {
        batch_iter->indices[i] = i;
    }
    return batch_iter;
}

void batch_iterator_free(BatchIterator *batch_iter) {
    free(batch_iter->indices);
    free(batch_iter);
}

void batch_iterator_shuffle(BatchIterator *batch_iter) {

    // Fisher-Yates shuffle
    for (int i = batch_iter->dataset->num_samples - 1; i >= 1; i--) {
        int j = rand_range(0, i);

        // Swap i & j
        int temp = batch_iter->indices[i];
        batch_iter->indices[i] = batch_iter->indices[j];
        batch_iter->indices[j] = temp;
    }
}

void batch_iterator_reset(BatchIterator *batch_iter) {
    batch_iter->current_idx = 0;
}

Batch *batch_iterator_next(BatchIterator *batch_iter) {
    const int N = batch_iter->dataset->num_samples;
    if (batch_iter->current_idx >= N) {
        return NULL;
    }

    // Batch bounds.
    int start = batch_iter->current_idx;
    int end = fmin(start + batch_iter->batch_size, N);
    int actual_size = end - start;

    // Allocate batch matrices.
    Batch *batch = (Batch *)malloc(sizeof(Batch));
    batch->X = matrix_create(actual_size, batch_iter->dataset->num_features);
    batch->Y =
        matrix_create(actual_size, batch_iter->dataset->Y->cols); // (actual_size, num_outputs)
    batch->size = actual_size;

    // Copy rows using shuffled indices with pre-allocated buffers
    Vector *row_buffer_x = vector_create(batch_iter->dataset->X->cols);
    Vector *row_buffer_y = vector_create(batch_iter->dataset->Y->cols);

    for (int i = 0; i < actual_size; i++) {
        int sample_idx = batch_iter->indices[start + i];
        matrix_copy_row_to_vector(row_buffer_x, batch_iter->dataset->X, sample_idx);
        matrix_copy_vector_into_row(batch->X, row_buffer_x, i);
        matrix_copy_row_to_vector(row_buffer_y, batch_iter->dataset->Y, sample_idx);
        matrix_copy_vector_into_row(batch->Y, row_buffer_y, i);
    }

    vector_free(row_buffer_x);
    vector_free(row_buffer_y);

    batch_iter->current_idx = end;
    return batch;
}
