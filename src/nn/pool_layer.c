/*
 * pool_layer.c
 *
 * Max pooling layer Implementations.
 */

// Lifecycle
#include "nn/pool_layer.h"
#include "tensor/tensor.h"
#include <math.h>
#include <stdlib.h>

MaxPoolLayer *maxpool_create(int pool_size, int stride) {
    MaxPoolLayer *layer = (MaxPoolLayer *)malloc(sizeof(MaxPoolLayer));
    layer->pool_size = pool_size;
    layer->stride = stride;
    layer->max_indices = NULL;
    return layer;
}

void maxpool_free(MaxPoolLayer *layer) {
    if (layer->max_indices != NULL) {
        free(layer->max_indices);
    }
    free(layer);
}

// Forward pass: returns output tensor, caches max_indices internally.
Tensor *maxpool_forward(MaxPoolLayer *layer, Tensor *input) {
    int C = input->shape[0];
    int H_in = input->shape[1];
    int W_in = input->shape[2];
    int H_out = (H_in - layer->pool_size) / layer->stride + 1;
    int W_out = (W_in - layer->pool_size) / layer->stride + 1;

    // Cache sizes for backward pass.
    layer->input_c = C;
    layer->input_h = H_in;
    layer->input_w = W_in;
    layer->output_h = H_out;
    layer->output_w = W_out;

    Tensor *Y = tensor_zeros(3, (int[]){C, H_out, W_out});
    layer->max_indices = (int *)malloc(sizeof(int) * C * H_out * W_out);

    for (int c = 0; c < C; c++) {
        for (int i = 0; i < H_out; i++) {
            for (int j = 0; j < W_out; j++) {
                // Top left corner of window of input.
                int h_start = i * layer->stride;
                int w_start = j * layer->stride;

                float max_val = -INFINITY;
                int max_idx = 0;

                for (int m = 0; m < layer->pool_size; m++) {
                    for (int n = 0; n < layer->pool_size; n++) {
                        float val = tensor_get3d(input, c, h_start + m, w_start + n);
                        if (val > max_val) {
                            max_val = val;
                            max_idx = encode_window_idx(m, n, layer->pool_size);
                        }
                    }
                }
                tensor_set3d(Y, c, i, j, max_val);
                layer->max_indices[out_idx(c, i, j, H_out, W_out)] = max_idx;
            }
        }
    }
    return Y;
}

// Backward pass: uses cached max_indices to route gradients.
Tensor *maxpool_backward(MaxPoolLayer *layer, Tensor *upstream_grad) {
    Tensor *dX = tensor_zeros(3, (int[]){layer->input_c, layer->input_h, layer->input_w});
    int H_out = upstream_grad->shape[1];
    int W_out = upstream_grad->shape[2];
    for (int c = 0; c < layer->input_c; c++) {
        for (int i = 0; i < H_out; i++) {
            for (int j = 0; j < W_out; j++) {
                int m, n;
                decode_window_index(layer->max_indices[out_idx(c, i, j, H_out, W_out)],
                                    layer->pool_size, &m, &n);

                int h_idx = i * layer->stride + m;
                int w_idx = j * layer->stride + n;
                dX->data[tensor_index3d(dX, c, h_idx, w_idx)] +=
                    tensor_get3d(upstream_grad, c, i, j);
            }
        }
    }
    return dX;
}
