/*
 * pool_layer.c
 *
 * Max pooling layer Implementations.
 */

// Lifecycle
#include "nn/pool_layer.h"

MaxPoolLayer *maxpool_create(int pool_size, int stride) {
}
void maxpool_free(MaxPoolLayer *layer) {
}

// Forward pass: returns output tensor, caches max_indices internally.
Tensor *maxpool_forward(MaxPoolLayer *layer, Tensor *input) {
}

// Backward pass: uses cached max_indices to route gradients.
Tensor *maxpool_backward(MaxPoolLayer *layer, Tensor *upstream_grad) {
}
