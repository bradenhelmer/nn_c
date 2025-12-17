/*
 * pool_layer.h
 *
 * Max pooling layer Declarations
 */

#ifndef POOL_LAYER_H
#define POOL_LAYER_H
#include "tensor/tensor.h"

typedef struct {
    int pool_size;
    int stride;

    // Cached during forward for backward pass.
    int input_c, input_h, input_w;
    int output_h, output_w;
    int *max_indices; // Flat array: [C * H_out * W_out]
} MaxPoolLayer;

// Lifecycle
MaxPoolLayer *maxpool_create(int pool_size, int stride);
void maxpool_free(MaxPoolLayer *layer);

// Forward pass: returns output tensor, caches max_indices internally.
Tensor *maxpool_forward(MaxPoolLayer *layer, Tensor *input);

// Backward pass: uses cached max_indices to route gradients.
Tensor *maxpool_backward(MaxPoolLayer *layer, Tensor *upstream_grad);

// Index helpers

// 3D -> 1D index for output/max_indices
static inline int out_idx(int c, int i, int j, int H_out, int W_out) {
    return c * (H_out * W_out) + i * W_out + j;
}

// Encode (m, n) -> flat window index
static inline int encode_window_idx(int m, int n, int pool_size) {
    return m * pool_size + n;
}

// Decode flat window index -> (m, n)
static inline void decode_window_index(int flat_idx, int pool_size, int *m, int *n) {
    *m = flat_idx / pool_size;
    *n = flat_idx % pool_size;
}

#endif /* ifndef POOL_LAYER_H*/
