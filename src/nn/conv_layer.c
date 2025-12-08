/*
 * conv_layer.c
 *
 * Convolutional Layer Implementations.
 */
#include "conv_layer.h"
#include <stdlib.h>

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

void conv_layer_init_weights(ConvLayer *layer);

// Forward/backward
Tensor *conv_layer_forward(ConvLayer *layer, Tensor *input);
Tensor *conv_layer_backward(ConvLayer *layer, Tensor *upstream_grad);
