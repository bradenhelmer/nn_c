/*
 * conv_layer.c
 *
 * Convolutional Layer Implementations.
 */
#include "layer.h"
#include "tensor/tensor.h"
#include "utils/utils.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

ConvLayer *conv_layer_create(int in_channels, int out_channels, int kernel_size, int stride,
                             int padding) {
    ConvLayer *cl = (ConvLayer *)malloc(sizeof(ConvLayer));
    cl->in_channels = in_channels;
    cl->out_channels = out_channels;
    cl->kernel_size = kernel_size;
    cl->stride = stride;
    cl->padding = padding;
    cl->kernels = tensor_create(4, (int[]){out_channels, in_channels, kernel_size, kernel_size});
    cl->biases = tensor_create(1, (int[]){out_channels});
    cl->input = NULL;
    cl->output = NULL;
    cl->d_kernels = tensor_create(4, (int[]){out_channels, in_channels, kernel_size, kernel_size});
    cl->d_biases = tensor_create(1, (int[]){out_channels});
    return cl;
}

void conv_layer_free(ConvLayer *layer) {
    tensor_free(layer->kernels);
    tensor_free(layer->biases);
    tensor_free(layer->d_kernels);
    tensor_free(layer->d_biases);
    if (layer->output != NULL) {
        tensor_free(layer->output);
    }
    if (layer->input != NULL) {
        tensor_free(layer->input);
    }
    free(layer);
}

// Xavier weight initialization
void conv_layer_init_weights(ConvLayer *layer) {
    int kernel_squared = layer->kernel_size * layer->kernel_size;
    int fan_in = layer->in_channels * kernel_squared;
    int fan_out = layer->out_channels * kernel_squared;
    float standard = sqrtf((2.f / (fan_in + fan_out)));
    for (int i = 0; i < layer->kernels->size; ++i) {
        layer->kernels->data[i] = rand_rangef(-standard, standard);
    }
}

Tensor *conv_layer_forward(ConvLayer *layer, Tensor *input) {
    Tensor *X_pad = tensor_pad2d(input, layer->padding);

    // Get output dimensions
    int H_out = (input->shape[1] - layer->kernel_size + 2 * layer->padding) / layer->stride + 1;
    int W_out = (input->shape[2] - layer->kernel_size + 2 * layer->padding) / layer->stride + 1;

    // Alloc output tensor
    Tensor *Y = tensor_zeros(3, (int[]){layer->out_channels, H_out, W_out});

    // Core loop
    for (int o = 0; o < layer->out_channels; o++) { // Each output channel
        for (int i = 0; i < H_out; i++) {           // Each output row
            for (int j = 0; j < W_out; j++) {       // Each output col
                float sum = layer->biases->data[o];
                for (int c = 0; c < layer->in_channels; c++) {         // Each input channel
                    for (int m = 0; m < layer->kernel_size; m++) {     // Kernel row
                        for (int n = 0; n < layer->kernel_size; n++) { // Kernel col
                            int h_idx = i * layer->stride + m;
                            int w_idx = j * layer->stride + n;
                            sum += tensor_get3d(X_pad, c, h_idx, w_idx) *
                                   tensor_get4d(layer->kernels, o, c, m, n);
                        }
                    }
                }
                tensor_set3d(Y, o, i, j, sum);
            }
        }
    }
    if (layer->input != NULL) {
        tensor_free(layer->input);
    }
    if (layer->output != NULL) {
        tensor_free(layer->output);
    }
    layer->input = X_pad; // Cache padded input for backward pass
    layer->output = Y;
    return Y;
}

Tensor *conv_layer_backward(ConvLayer *layer, Tensor *upstream_grad) {

    int H_out =
        (layer->input->shape[1] - layer->kernel_size + 2 * layer->padding) / layer->stride + 1;
    int W_out =
        (layer->input->shape[2] - layer->kernel_size + 2 * layer->padding) / layer->stride + 1;

    // 1. Gradient w.r.t biases
    // Each bias[out_channel] is added to every spatial position in out_channel. The gradient
    // is the sum of upstream gradients over the spatial dimensions.
    for (int out_channel = 0; out_channel < layer->out_channels; out_channel++) {
        float sum = 0.f;
        for (int i = 0; i < H_out; i++) {
            for (int j = 0; j < W_out; j++) {
                sum += tensor_get3d(upstream_grad, out_channel, i, j);
            }
        }
        layer->d_biases->data[out_channel] = sum;
    }

    // 2. Gradient w.r.t Kernels
    // Kernel element K[out_channel, in_channel, m, n] computes every output in channel o,
    // being multiplied by a different input element each time.
    //
    // From forward pass:
    // Y[o,i,j] = ... + X_pad[c,i*s+m,j*s+n] * K[o,c,m,n] + ...
    //
    // By chain rule, K[o,c,m,n] affects Y[o,i,j] with local gradient X_pad[c,i*s+m,j*s+n]
    //
    //     ∂L
    // ----------- = (0..H_out-1)∑ (0..W_out-1)∑ δ[o,i,j] * X_pad[c,i*s+m,j*s+n]
    // ∂K[o,c,m,n]
    //
    // This is convolution of the input with the upstream gradient.

    for (int o = 0; o < layer->out_channels; o++) {            // Each output channel
        for (int c = 0; c < layer->in_channels; c++) {         // Each input channel
            for (int m = 0; m < layer->kernel_size; m++) {     // Kernel row
                for (int n = 0; n < layer->kernel_size; n++) { // Kernel col
                    float sum = 0.0f;
                    for (int i = 0; i < H_out; i++) {     // Each output row
                        for (int j = 0; j < W_out; j++) { // Each output col
                            int h_idx = i * layer->stride + m;
                            int w_idx = j * layer->stride + n;
                            sum += tensor_get3d(upstream_grad, o, i, j) *
                                   tensor_get3d(layer->input, c, h_idx, w_idx);
                        }
                    }
                    tensor_set4d(layer->d_kernels, o, c, m, n, sum);
                }
            }
        }
    }

    // 3. Gradient w.r.t input
    // Each input element X_pad[c, h, w] contributes to multiple output elements,
    // every position where the kernel "covers" that input location.
    //
    // From forward pass:
    // Y[o,i,j] = ... + X_pad[c,i*s+m,j*s+n] * K[o,c,m,n] + ...
    //
    // Rearranging: X_pad[c, h, w] affects Y[o, i, j] when:
    // - h = i * s + m, i.e., i = (h - m) / s
    // - w = j * s + n, i.e., j = (w - n) / s
    // and the local gradient is K[o,c,m,n].
    //
    //       ∂L                                                             (h - m)  (w - n)
    // ------------- = (0..C_out-1)∑(0..kh-1)∑(0..kw-1)∑ K[o,c,m,n] * δ [o, -------, -------]
    // ∂X_pad[c,h,w]                                                           s        s
    //
    // But only when (h - m) and (w - n) are divisible by stride and the result
    // is in valid indices.
    //
    // Assuming stride == 1 for now.
    //
    Tensor *dX_pad = tensor_zeros(3, layer->input->shape);
    int H_padded = dX_pad->shape[1];
    int W_padded = dX_pad->shape[2];
    for (int c = 0; c < layer->in_channels; c++) { // Each input channel
        for (int h = 0; h < H_padded; h++) {
            for (int w = 0; w < W_padded; w++) {
                float sum = 0.0f;
                for (int o = 0; o < layer->out_channels; o++) {
                    for (int m = 0; m < layer->kernel_size; m++) {     // Kernel row
                        for (int n = 0; n < layer->kernel_size; n++) { // Kernel col
                            int i = h - m;
                            int j = w - n;
                            if ((i >= 0 && i < H_out) && (j >= 0 && j < W_out)) {
                                sum += tensor_get4d(layer->kernels, o, c, m, n) *
                                       tensor_get3d(upstream_grad, o, i, j);
                            }
                        }
                    }
                }
                tensor_set3d(dX_pad, c, h, w, sum);
            }
        }
    }
    Tensor *dX = tensor_unpad2d(dX_pad, layer->padding);
    tensor_free(dX_pad);
    return dX;
}
