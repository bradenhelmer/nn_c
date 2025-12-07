/*
 * test_batch.c - Comprehensive tests for batch operations
 */

#include "../src/data/batch.h"
#include "../src/data/dataset.h"
#include "test_runner.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

// =============================================================================
// Creation and Destruction Tests
// =============================================================================

void test_batch_iterator_create() {
    Dataset *data = create_xor_gate_dataset();
    BatchIterator *iter = batch_iterator_create(data, 2);

    assert(iter != NULL);
    assert(iter->dataset == data);
    assert(iter->batch_size == 2);
    assert(iter->num_batches == 2); // 4 samples / 2 batch_size = 2 batches
    assert(iter->current_idx == 0);
    assert(iter->indices != NULL);

    batch_iterator_free(iter);
    dataset_free(data);
    TEST_PASSED;
}

void test_batch_iterator_create_uneven_batches() {
    Dataset *data = create_xor_gate_dataset();
    BatchIterator *iter = batch_iterator_create(data, 3);

    assert(iter != NULL);
    assert(iter->batch_size == 3);
    assert(iter->num_batches == 2); // ceil(4 / 3) = 2 batches

    batch_iterator_free(iter);
    dataset_free(data);
    TEST_PASSED;
}

void test_batch_iterator_create_single_batch() {
    Dataset *data = create_xor_gate_dataset();
    BatchIterator *iter = batch_iterator_create(data, 10);

    assert(iter != NULL);
    assert(iter->num_batches == 1); // All data in one batch

    batch_iterator_free(iter);
    dataset_free(data);
    TEST_PASSED;
}

// =============================================================================
// Iterator Reset Tests
// =============================================================================

void test_batch_iterator_reset() {
    Dataset *data = create_and_gate_dataset();
    BatchIterator *iter = batch_iterator_create(data, 2);

    // Advance the iterator
    iter->current_idx = 3;

    // Reset should bring current_idx back to 0
    batch_iterator_reset(iter);
    assert(iter->current_idx == 0);

    batch_iterator_free(iter);
    dataset_free(data);
    TEST_PASSED;
}

// =============================================================================
// Batch Iterator Next Tests
// =============================================================================

void test_batch_iterator_next_basic() {
    Dataset *data = create_xor_gate_dataset();
    BatchIterator *iter = batch_iterator_create(data, 2);

    // Initialize indices to 0,1,2,3 (no shuffle for predictable test)
    for (int i = 0; i < data->num_samples; i++) {
        iter->indices[i] = i;
    }

    // Get first batch
    Batch *batch1 = batch_iterator_next(iter);
    assert(batch1 != NULL);
    assert(batch1->X->rows == 2);
    assert(batch1->X->cols == 2);
    assert(batch1->Y->rows == 2);
    assert(batch1->Y->cols == 1);

    // Get second batch
    Batch *batch2 = batch_iterator_next(iter);
    assert(batch2 != NULL);
    assert(batch2->X->rows == 2);

    // Try to get third batch (should be NULL)
    Batch *batch3 = batch_iterator_next(iter);
    assert(batch3 == NULL);

    batch_free(batch1);
    batch_free(batch2);
    batch_iterator_free(iter);
    dataset_free(data);
    TEST_PASSED;
}

void test_batch_iterator_next_uneven_last_batch() {
    Dataset *data = create_xor_gate_dataset();
    BatchIterator *iter = batch_iterator_create(data, 3);

    // Initialize indices
    for (int i = 0; i < data->num_samples; i++) {
        iter->indices[i] = i;
    }

    // Get first batch (size 3)
    Batch *batch1 = batch_iterator_next(iter);
    assert(batch1 != NULL);
    assert(batch1->X->rows == 3);

    // Get second batch (size 1, the remainder)
    Batch *batch2 = batch_iterator_next(iter);
    assert(batch2 != NULL);
    assert(batch2->X->rows == 1);

    // No more batches
    Batch *batch3 = batch_iterator_next(iter);
    assert(batch3 == NULL);

    batch_free(batch1);
    batch_free(batch2);
    batch_iterator_free(iter);
    dataset_free(data);
    TEST_PASSED;
}

void test_batch_iterator_next_exhausted() {
    Dataset *data = create_and_gate_dataset();
    BatchIterator *iter = batch_iterator_create(data, 2);

    // Initialize indices
    for (int i = 0; i < data->num_samples; i++) {
        iter->indices[i] = i;
    }

    // Exhaust the iterator
    Batch *b1 = batch_iterator_next(iter);
    Batch *b2 = batch_iterator_next(iter);
    batch_free(b1);
    batch_free(b2);

    // Should return NULL when exhausted
    Batch *b3 = batch_iterator_next(iter);
    assert(b3 == NULL);

    // Should still return NULL
    Batch *b4 = batch_iterator_next(iter);
    assert(b4 == NULL);

    batch_iterator_free(iter);
    dataset_free(data);
    TEST_PASSED;
}

void test_batch_iterator_reset_and_reuse() {
    Dataset *data = create_or_gate_dataset();
    BatchIterator *iter = batch_iterator_create(data, 2);

    // Initialize indices
    for (int i = 0; i < data->num_samples; i++) {
        iter->indices[i] = i;
    }

    // First pass
    Batch *b1 = batch_iterator_next(iter);
    Batch *b2 = batch_iterator_next(iter);
    assert(b1 != NULL);
    assert(b2 != NULL);
    batch_free(b1);
    batch_free(b2);

    // Reset and use again
    batch_iterator_reset(iter);
    Batch *b3 = batch_iterator_next(iter);
    Batch *b4 = batch_iterator_next(iter);
    assert(b3 != NULL);
    assert(b4 != NULL);
    batch_free(b3);
    batch_free(b4);

    batch_iterator_free(iter);
    dataset_free(data);
    TEST_PASSED;
}

// =============================================================================
// Batch Content Verification Tests
// =============================================================================

void test_batch_content_integrity() {
    Dataset *data = create_xor_gate_dataset();
    BatchIterator *iter = batch_iterator_create(data, 2);

    // Initialize indices to 0,1 for predictable test
    for (int i = 0; i < data->num_samples; i++) {
        iter->indices[i] = i;
    }

    // Get first batch (should contain samples 0 and 1)
    Batch *batch = batch_iterator_next(iter);
    assert(batch != NULL);

    // Verify dimensions
    assert(batch->X->rows == 2);
    assert(batch->X->cols == data->num_features);
    assert(batch->Y->rows == 2);
    assert(batch->Y->cols == data->Y->cols);

    batch_free(batch);
    batch_iterator_free(iter);
    dataset_free(data);
    TEST_PASSED;
}

void test_batch_single_sample() {
    Dataset *data = create_and_gate_dataset();
    BatchIterator *iter = batch_iterator_create(data, 1);

    // Initialize indices
    for (int i = 0; i < data->num_samples; i++) {
        iter->indices[i] = i;
    }

    // Should get 4 batches of size 1
    for (int i = 0; i < 4; i++) {
        Batch *batch = batch_iterator_next(iter);
        assert(batch != NULL);
        assert(batch->X->rows == 1);
        assert(batch->Y->rows == 1);
        batch_free(batch);
    }

    // No more batches
    Batch *last = batch_iterator_next(iter);
    assert(last == NULL);

    batch_iterator_free(iter);
    dataset_free(data);
    TEST_PASSED;
}

void test_batch_full_dataset() {
    Dataset *data = create_xor_gate_dataset();
    BatchIterator *iter = batch_iterator_create(data, 4);

    // Initialize indices
    for (int i = 0; i < data->num_samples; i++) {
        iter->indices[i] = i;
    }

    // Should get exactly 1 batch containing all samples
    Batch *batch = batch_iterator_next(iter);
    assert(batch != NULL);
    assert(batch->X->rows == 4);
    assert(batch->Y->rows == 4);

    // No more batches
    Batch *next = batch_iterator_next(iter);
    assert(next == NULL);

    batch_free(batch);
    batch_iterator_free(iter);
    dataset_free(data);
    TEST_PASSED;
}

// =============================================================================
// Shuffle Tests
// =============================================================================

void test_batch_iterator_shuffle() {
    Dataset *data = create_xor_gate_dataset();
    BatchIterator *iter = batch_iterator_create(data, 2);

    // Initialize indices in order
    for (int i = 0; i < data->num_samples; i++) {
        iter->indices[i] = i;
    }

    // Store original order
    int original[4];
    for (int i = 0; i < 4; i++) {
        original[i] = iter->indices[i];
    }

    // Shuffle
    batch_iterator_shuffle(iter);

    // All indices should still be present (no duplicates/missing values)
    int found[4] = {0};
    for (int i = 0; i < 4; i++) {
        assert(iter->indices[i] >= 0 && iter->indices[i] < 4);
        found[iter->indices[i]]++;
    }

    // Each index should appear exactly once
    for (int i = 0; i < 4; i++) {
        assert(found[i] == 1);
    }

    batch_iterator_free(iter);
    dataset_free(data);
    TEST_PASSED;
}

// =============================================================================
// Test Runner
// =============================================================================

void run_batch_tests(void) {
    printf("\n=== Batch Iterator Creation and Destruction Tests ===\n");
    test_batch_iterator_create();
    test_batch_iterator_create_uneven_batches();
    test_batch_iterator_create_single_batch();

    printf("\n=== Batch Iterator Reset Tests ===\n");
    test_batch_iterator_reset();

    printf("\n=== Batch Iterator Next Tests ===\n");
    test_batch_iterator_next_basic();
    test_batch_iterator_next_uneven_last_batch();
    test_batch_iterator_next_exhausted();
    test_batch_iterator_reset_and_reuse();

    printf("\n=== Batch Content Verification Tests ===\n");
    test_batch_content_integrity();
    test_batch_single_sample();
    test_batch_full_dataset();

    printf("\n=== Batch Shuffle Tests ===\n");
    test_batch_iterator_shuffle();

    printf("\n=== All Batch Tests Passed! ===\n");
}
