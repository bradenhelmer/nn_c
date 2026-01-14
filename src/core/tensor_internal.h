/*
 * tensor_internal.h - Internal tensor struct declaration.
 */

#ifndef TENSOR_INTERNAL_H
#define TENSOR_INTERNAL_H

#include "core/tensor.h"

struct Tensor {
    float *data;
    int *shape;   // e.g {32, 28, 28} for 32 channels of 28x28
    int *strides; // precomputed to faster indexing
    int ndim;     // Number of dimensions
    int size;     // Total elements
    int owner;    // Does this tensor own the data? For views
};

#endif /* ifndef TENSOR_INTERNAL_H */
