/*
 * batch.h
 *
 * Batch training declarations.
 */
#ifndef BATCH_H
#define BATCH_H
#include "dataset.h"

typedef struct {
    Matrix *X; // (batch_size, num_features)
    Matrix *Y; // (batch_size, num_outputs)
    int size;
} Batch;

void batch_free(Batch *b);

typedef struct {
    Dataset *dataset; // REFERENCE POINTER
    int batch_size;
    int num_batches;
    int current_idx;
    int *indices; // Shuffled sample indices
} BatchIterator;

BatchIterator *batch_iterator_create(Dataset *data, int batch_size);
void batch_iterator_free(BatchIterator *batch_iter);
void batch_iterator_shuffle(BatchIterator *batch_iter);
void batch_iterator_reset(BatchIterator *batch_iter);  // Reset without shuffle
Batch *batch_iterator_next(BatchIterator *batch_iter); // NULL when exhausted

#endif /* ifndef BATCH_H*/
