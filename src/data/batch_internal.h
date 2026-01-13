/*
 * batch_internal.h - Internal Batch and BatchIterator struct definitions
 */

#ifndef BATCH_INTERNAL_H
#define BATCH_INTERNAL_H
#include "batch.h"
#include "dataset.h"
#include "tensor/tensor.h"

// Represents a single mini-batch extracted from a dataset
// NOTE: Batch OWNS its X and Y tensors - batch_free() will free them
struct Batch {
    Tensor *X; // 2D: (batch_size, num_features) - owned, will be freed
    Tensor *Y; // 2D: (batch_size, num_outputs) - owned, will be freed
    int size;  // Actual number of samples in this batch (may be < batch_size for last batch)
};

// Iterator for traversing a dataset in batches
// NOTE: BatchIterator does NOT own the dataset - just borrows a reference
struct BatchIterator {
    Dataset *dataset; // Borrowed reference - do not free
    int batch_size;   // Requested batch size
    int num_batches;  // Total number of batches
    int current_idx;  // Current position in iteration
    int *indices;     // Owned array of shuffled sample indices - freed by batch_iterator_free()
};

#endif /* ifndef BATCH_INTERNAL_H */
