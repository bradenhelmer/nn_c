/*
 * dataset.h
 *
 * Struct and function declarations for dataset handling.
 */
#ifndef DATASET_H
#define DATASET_H
#include "../linalg/matrix.h"
#include "../linalg/vector.h"

typedef struct {
    Matrix *X;
    Vector *y;
    int num_samples;
    int num_features;
} Dataset;

Dataset *dataset_create(int num_samples, int num_features);
void dataset_free(Dataset *d);

// Test datasets
Dataset *create_and_gate_dataset();
Dataset *create_or_gate_dataset();
Dataset *create_xor_dataset();

// Data splitting
void dataset_shuffle(Dataset *d);
void dataset_split(Dataset *source, Dataset *train, Dataset *test, float train_ratio);

#endif /* ifndef DATASET_H */
