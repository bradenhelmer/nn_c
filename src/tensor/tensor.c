/*
 * tensor.c
 *
 * Tensor implementations.
 */
#include "tensor.h"
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

Tensor *tensor_create(int ndim, int *shape) {
    Tensor *t = (Tensor *)malloc(sizeof(Tensor));
    t->ndim = ndim;

    t->shape = (int *)malloc(ndim * sizeof(int));
    memcpy(t->shape, shape, ndim * sizeof(int));

    // Compute strides left to right
    t->strides = (int *)malloc(ndim * sizeof(int));
    t->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        t->strides[i] = t->strides[i + 1] * shape[i + 1];
    }

    t->size = t->strides[0] * shape[0];
    t->data = (float *)malloc(t->size * sizeof(float));
    return t;
}

Tensor *tensor_zeros(int ndim, int *shape) {
    return tensor_create(ndim, shape);
}

void tensor_free(Tensor *t) {
    free(t->data);
    free(t->shape);
    free(t->strides);
    free(t);
}

float tensor_get3d(Tensor *t, int i, int j, int k) {
    return t->data[tensor_index3d(t, i, j, k)];
}

void tensor_set3d(Tensor *t, int i, int j, int k, float value) {
    t->data[tensor_index3d(t, i, j, k)] = value;
}

float tensor_get4d(Tensor *t, int i, int j, int k, int l) {
    return t->data[tensor_index4d(t, i, j, k, l)];
}

void tensor_set4d(Tensor *t, int i, int j, int k, int l, float value) {
    t->data[tensor_index4d(t, i, j, k, l)] = value;
}

void tensor_fill(Tensor *t, float val) {
    for (int i = 0; i < t->size; i++) {
        t->data[i] = val;
    }
}

void tensor_copy(Tensor *dest, const Tensor *src) {
    assert(dest->ndim == src->ndim);
    for (int shape = 0; shape < dest->ndim; shape++) {
        assert(dest->shape[shape] == src->shape[shape]);
    }
    memcpy(dest->data, src->data, sizeof(float) * src->size);
}

Tensor *tensor_clone(const Tensor *t) {
    Tensor *clone = tensor_create(t->ndim, t->shape);
    memcpy(clone->data, t->data, sizeof(float) * t->size);
    return clone;
}

void tensor_print_shape(const Tensor *t) {
    int i;
    printf("Shape: {");
    for (i = 0; i < t->ndim - 1; i++) {
        printf("%d, ", t->shape[i]);
    }
    printf("%d}", t->shape[i]);
}

Tensor *tensor_pad2d(const Tensor *t, int padding);
