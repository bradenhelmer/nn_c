/*
 * batch.h
 *
 * Batch training declarations.
 */
#ifndef BATCH_H
#define BATCH_H
#include "dataset.h"

// Opaque types - see batch_internal.h for struct definitions
typedef struct Batch Batch;
typedef struct BatchIterator BatchIterator;

// Frees the batch and its owned tensors (X and Y)
void batch_free(Batch *b);

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
