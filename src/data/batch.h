/*
 * batch.h
 *
 * Batch training declarations.
 */
#ifndef BATCH_H
#define BATCH_H
#include "dataset.h"

// Represents a single mini-batch extracted from a dataset
// NOTE: Batch OWNS its X and Y tensors - batch_free() will free them
typedef struct {
    Tensor *X; // 2D: (batch_size, num_features) - owned, will be freed
    Tensor *Y; // 2D: (batch_size, num_outputs) - owned, will be freed
    int size;  // Actual number of samples in this batch (may be < batch_size for last batch)
} Batch;

// Frees the batch and its owned tensors (X and Y)
void batch_free(Batch *b);

// Iterator for traversing a dataset in batches
// NOTE: BatchIterator does NOT own the dataset - just borrows a reference
typedef struct {
    Dataset *dataset; // Borrowed reference - do not free
    int batch_size;   // Requested batch size
    int num_batches;  // Total number of batches
    int current_idx;  // Current position in iteration
    int *indices;     // Owned array of shuffled sample indices - freed by batch_iterator_free()
} BatchIterator;

// Creates a new batch iterator over the given dataset
// Does NOT take ownership of dataset - dataset must outlive the iterator
BatchIterator *batch_iterator_create(Dataset *data, int batch_size);

// Frees the iterator and its owned resources (indices array)
// Does NOT free the dataset (borrowed reference)
void batch_iterator_free(BatchIterator *batch_iter);

void batch_iterator_shuffle(BatchIterator *batch_iter);
void batch_iterator_reset(BatchIterator *batch_iter); // Reset without shuffle

// Returns next batch from the iterator, or NULL when exhausted
// Caller MUST free the returned Batch with batch_free() when done
Batch *batch_iterator_next(BatchIterator *batch_iter);

#endif /* ifndef BATCH_H*/
