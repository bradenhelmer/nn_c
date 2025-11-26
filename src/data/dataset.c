/*
 * dataset.c
 *
 * Dataset function implementations.
 */
#include "dataset.h"
#include <stdlib.h>

Dataset *dataset_create(int num_samples, int num_features) {
    Dataset *d = (Dataset *)malloc(sizeof(Dataset));
    d->num_samples = num_samples;
    d->num_features = num_features;
    return d;
}
void dataset_free(Dataset *d) {
    matrix_free(d->X);
    vector_free(d->y);
    free(d);
}

static Matrix *_create_2bit_input_matrix() {
    Matrix *m = matrix_zeros(4, 2);
    matrix_set(m, 1, 1, 1);
    matrix_set(m, 2, 0, 1);
    matrix_set(m, 3, 0, 1);
    matrix_set(m, 3, 1, 1);
    return m;
}

// Test datasets
Dataset *create_and_gate_dataset() {
    Dataset *d = dataset_create(4, 2);
    d->X = _create_2bit_input_matrix();
    d->y = vector_zeros(4);
    d->y->data[3] = 1;
    return d;
}

Dataset *create_or_gate_dataset() {
    Dataset *d = dataset_create(4, 2);
    d->X = _create_2bit_input_matrix();
    d->y = vector_ones(4);
    d->y->data[0] = 0;
    return d;
}

Dataset *create_xor_gate_dataset() {
    Dataset *d = dataset_create(4, 2);
    d->X = _create_2bit_input_matrix();
    d->y = vector_zeros(4);
    d->y->data[1] = 1;
    d->y->data[2] = 1;
    return d;
}
