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

Layer *conv_layer_create(int in_channels, int out_channels, int kernel_size, int stride,
                         int padding) {
    ConvLayer *cl = (ConvLayer *)malloc(sizeof(ConvLayer));
    cl->in_channels = in_channels;
    cl->out_channels = out_channels;
    cl->kernel_size = kernel_size;
    cl->stride = stride;
    cl->padding = padding;
    cl->weights = tensor_create4d(out_channels, in_channels, kernel_size, kernel_size);
    cl->biases = tensor_create1d(out_channels);
    cl->input = NULL;
    cl->output = NULL;
    cl->grad_weights = tensor_create4d(out_channels, in_channels, kernel_size, kernel_size);
    cl->grad_biases = tensor_create1d(out_channels);
    conv_layer_init_weights(cl);
    return layer_create(LAYER_CONV_2D, (void *)cl);
}

void conv_layer_free(ConvLayer *layer) {
    tensor_free(layer->weights);
    tensor_free(layer->biases);
    tensor_free(layer->grad_weights);
    tensor_free(layer->grad_biases);
    if (layer->output != NULL) {
        tensor_free(layer->output);
    }
    if (layer->input != NULL) {
        tensor_free(layer->input);
    }
    free(layer);
}

// Compute convolution parameters from layer and input dimensions
ConvParams conv_params_create(const ConvLayer *layer, const Tensor *input) {
    ConvParams p;
    p.C_in = layer->in_channels;
    p.C_out = layer->out_channels;
    p.K = layer->kernel_size;
    p.stride = layer->stride;
    p.padding = layer->padding;

    p.H_in = input->shape[1];
    p.W_in = input->shape[2];
    p.H_padded = p.H_in + 2 * p.padding;
    p.W_padded = p.W_in + 2 * p.padding;
    p.H_out = (p.H_padded - p.K) / p.stride + 1;
    p.W_out = (p.W_padded - p.K) / p.stride + 1;

    return p;
}

// Compute convolution parameters from layer and already-padded input
ConvParams conv_params_from_padded(const ConvLayer *layer, const Tensor *padded_input) {
    ConvParams p;
    p.C_in = layer->in_channels;
    p.C_out = layer->out_channels;
    p.K = layer->kernel_size;
    p.stride = layer->stride;
    p.padding = layer->padding;

    p.H_padded = padded_input->shape[1];
    p.W_padded = padded_input->shape[2];
    p.H_in = p.H_padded - 2 * p.padding;
    p.W_in = p.W_padded - 2 * p.padding;
    p.H_out = (p.H_padded - p.K) / p.stride + 1;
    p.W_out = (p.W_padded - p.K) / p.stride + 1;

    return p;
}

// Xavier weight initialization
void conv_layer_init_weights(ConvLayer *layer) {
    int kernel_squared = layer->kernel_size * layer->kernel_size;
    int fan_in = layer->in_channels * kernel_squared;
    int fan_out = layer->out_channels * kernel_squared;
    float standard = sqrtf((2.f / (fan_in + fan_out)));
    for (int i = 0; i < layer->weights->size; ++i) {
        layer->weights->data[i] = rand_rangef(-standard, standard);
    }
}

Tensor *conv_layer_forward(ConvLayer *layer, const Tensor *input) {
    ConvParams p = conv_params_create(layer, input);
    Tensor *X_pad = tensor_pad2d(input, p.padding);
    Tensor *Y = tensor_create3d(p.C_out, p.H_out, p.W_out);

    // Core loop
    for (int o = 0; o < p.C_out; o++) {         // Each output channel
        for (int i = 0; i < p.H_out; i++) {     // Each output row
            for (int j = 0; j < p.W_out; j++) { // Each output col
                float sum = layer->biases->data[o];
                for (int c = 0; c < p.C_in; c++) {      // Each input channel
                    for (int m = 0; m < p.K; m++) {     // Kernel row
                        for (int n = 0; n < p.K; n++) { // Kernel col
                            int h_idx = i * p.stride + m;
                            int w_idx = j * p.stride + n;
                            sum += tensor_get3d(X_pad, c, h_idx, w_idx) *
                                   tensor_get4d(layer->weights, o, c, m, n);
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
    layer->input = X_pad;
    layer->output = Y;
    return Y;
}

Tensor *conv_layer_forward_stride_optimized(ConvLayer *layer, const Tensor *input) {
    ConvParams p = conv_params_create(layer, input);
    Tensor *X_pad = tensor_pad2d(input, p.padding);

    // Fetch padded input metadata once
    float *X_pad_data = X_pad->data;
    int X_pad_stride_c = X_pad->strides[0]; // H_padded * W_padded
    int X_pad_stride_h = X_pad->strides[1]; // W_padded

    // Fetch weight tensor metadata once
    float *W_data = layer->weights->data;
    int W_stride_out = layer->weights->strides[0]; // C_in * K * K
    int W_stride_in = layer->weights->strides[1];  // K * K
    int W_stride_kh = layer->weights->strides[2];  // K

    // Create output tensor and fetch metadata once
    Tensor *Y = tensor_create3d(p.C_out, p.H_out, p.W_out);
    float *Y_data = Y->data;
    int Y_stride_c = Y->strides[0]; // H_out * W_out
    int Y_stride_h = Y->strides[1]; // W_out

    // Core computation with direct addressing
    for (int out_c = 0; out_c < p.C_out; out_c++) {
        float *W_out_base = W_data + out_c * W_stride_out;
        float *Y_out_base = Y_data + out_c * Y_stride_c;

        for (int out_h = 0; out_h < p.H_out; out_h++) {
            float *Y_row_ptr = Y_out_base + out_h * Y_stride_h;
            for (int out_w = 0; out_w < p.W_out; out_w++) {
                float sum = layer->biases->data[out_c];
                int h_start = out_h * p.stride;
                int w_start = out_w * p.stride;

                for (int in_c = 0; in_c < p.C_in; in_c++) {
                    float *W_kernel_base = W_out_base + in_c * W_stride_in;
                    float *X_pad_channel_base = X_pad_data + in_c * X_pad_stride_c;

                    for (int kh = 0; kh < p.K; kh++) {
                        int h_in = h_start + kh;
                        float *X_row_ptr = X_pad_channel_base + h_in * X_pad_stride_h;
                        float *W_row_ptr = W_kernel_base + kh * W_stride_kh;

                        for (int kw = 0; kw < p.K; kw++) {
                            int w_in = w_start + kw;
                            sum += X_row_ptr[w_in] * W_row_ptr[kw];
                        }
                    }
                }
                *Y_row_ptr++ = sum;
            }
        }
    }

    if (layer->input != NULL) {
        tensor_free(layer->input);
    }
    if (layer->output != NULL) {
        tensor_free(layer->output);
    }
    layer->input = X_pad;
    layer->output = Y;
    return Y;
}

Tensor *conv_layer_backward(ConvLayer *layer, const Tensor *upstream_grad) {
    // Create params from cached padded input (layer->input is already padded)
    ConvParams p = conv_params_from_padded(layer, layer->input);

    // 1. Gradient w.r.t biases
    for (int out_c = 0; out_c < p.C_out; out_c++) {
        float sum = 0.0f;
        for (int i = 0; i < p.H_out; i++) {
            for (int j = 0; j < p.W_out; j++) {
                sum += tensor_get3d(upstream_grad, out_c, i, j);
            }
        }
        layer->grad_biases->data[out_c] = sum;
    }

    // 2. Gradient w.r.t Kernels
    for (int o = 0; o < p.C_out; o++) {
        for (int c = 0; c < p.C_in; c++) {
            for (int m = 0; m < p.K; m++) {
                for (int n = 0; n < p.K; n++) {
                    float sum = 0.0f;
                    for (int i = 0; i < p.H_out; i++) {
                        for (int j = 0; j < p.W_out; j++) {
                            int h_idx = i * p.stride + m;
                            int w_idx = j * p.stride + n;
                            sum += tensor_get3d(upstream_grad, o, i, j) *
                                   tensor_get3d(layer->input, c, h_idx, w_idx);
                        }
                    }
                    tensor_set4d(layer->grad_weights, o, c, m, n, sum);
                }
            }
        }
    }

    // 3. Gradient w.r.t input
    Tensor *dX_pad = tensor_create3d(p.C_in, p.H_padded, p.W_padded);
    for (int c = 0; c < p.C_in; c++) {
        for (int h = 0; h < p.H_padded; h++) {
            for (int w = 0; w < p.W_padded; w++) {
                float sum = 0.0f;
                for (int o = 0; o < p.C_out; o++) {
                    for (int m = 0; m < p.K; m++) {
                        for (int n = 0; n < p.K; n++) {
                            int i = h - m;
                            int j = w - n;
                            if ((i >= 0 && i < p.H_out) && (j >= 0 && j < p.W_out)) {
                                sum += tensor_get4d(layer->weights, o, c, m, n) *
                                       tensor_get3d(upstream_grad, o, i, j);
                            }
                        }
                    }
                }
                tensor_set3d(dX_pad, c, h, w, sum);
            }
        }
    }

    Tensor *dX = tensor_unpad2d(dX_pad, p.padding);
    tensor_free(dX_pad);
    return dX;
}

Tensor *conv_layer_backward_stride_optimized(ConvLayer *layer, const Tensor *upstream_grad) {
    // Create params from cached padded input
    ConvParams p = conv_params_from_padded(layer, layer->input);

    // Fetch upstream gradient metadata once
    float *UG_data = upstream_grad->data;
    int UG_stride_c = upstream_grad->strides[0]; // H_out * W_out
    int UG_stride_h = upstream_grad->strides[1]; // W_out

    // Fetch weight gradient metadata once
    float *GW_data = layer->grad_weights->data;
    int GW_stride_out = layer->grad_weights->strides[0]; // C_in * K * K
    int GW_stride_in = layer->grad_weights->strides[1];  // K * K
    int GW_stride_kh = layer->grad_weights->strides[2];  // K

    // Fetch input (padded) tensor metadata once
    float *X_data = layer->input->data;
    int X_stride_c = layer->input->strides[0]; // H_padded * W_padded
    int X_stride_h = layer->input->strides[1]; // W_padded

    // 1. Gradient w.r.t biases
    float *grad_bias_base = layer->grad_biases->data;
    for (int out_c = 0; out_c < p.C_out; out_c++) {
        float *UG_out_base = UG_data + out_c * UG_stride_c;
        float sum = 0.0f;
        for (int out_h = 0; out_h < p.H_out; out_h++) {
            float *UG_row_ptr = UG_out_base + out_h * UG_stride_h;
            for (int out_w = 0; out_w < p.W_out; out_w++) {
                sum += UG_row_ptr[out_w];
            }
        }
        grad_bias_base[out_c] = sum;
    }

    // 2. Gradient w.r.t kernels (weights)
    for (int out_c = 0; out_c < p.C_out; out_c++) {
        float *UG_out_base = UG_data + out_c * UG_stride_c;
        float *GW_out_base = GW_data + out_c * GW_stride_out;
        for (int in_c = 0; in_c < p.C_in; in_c++) {
            float *GW_in_base = GW_out_base + in_c * GW_stride_in;
            float *X_in_base = X_data + in_c * X_stride_c;
            for (int kh = 0; kh < p.K; kh++) {
                float *GW_row_ptr = GW_in_base + kh * GW_stride_kh;
                for (int kw = 0; kw < p.K; kw++) {
                    float sum = 0.0f;
                    for (int out_h = 0; out_h < p.H_out; out_h++) {
                        float *UG_row_ptr = UG_out_base + out_h * UG_stride_h;
                        float *X_row_ptr = X_in_base + (out_h * p.stride + kh) * X_stride_h;
                        for (int out_w = 0; out_w < p.W_out; out_w++) {
                            int w_idx = out_w * p.stride + kw;
                            sum += UG_row_ptr[out_w] * X_row_ptr[w_idx];
                        }
                    }
                    *GW_row_ptr++ = sum;
                }
            }
        }
    }

    // 3. Gradient w.r.t input
    Tensor *dX_pad = tensor_create3d(p.C_in, p.H_padded, p.W_padded);
    float *dX_pad_data = dX_pad->data;
    int dX_pad_stride_in = dX_pad->strides[0]; // H_padded * W_padded
    int dX_pad_stride_h = dX_pad->strides[1];  // W_padded

    // Fetch weight tensor metadata once
    float *W_data = layer->weights->data;
    int W_stride_out = layer->weights->strides[0]; // C_in * K * K
    int W_stride_in = layer->weights->strides[1];  // K * K
    int W_stride_kh = layer->weights->strides[2];  // K

    for (int in_c = 0; in_c < p.C_in; in_c++) {
        float *dX_pad_in_base = dX_pad_data + in_c * dX_pad_stride_in;
        for (int h_in = 0; h_in < p.H_padded; h_in++) {
            float *dX_pad_row_ptr = dX_pad_in_base + h_in * dX_pad_stride_h;
            for (int w_in = 0; w_in < p.W_padded; w_in++) {
                float sum = 0.0f;
                for (int out_c = 0; out_c < p.C_out; out_c++) {
                    float *W_in_base = W_data + out_c * W_stride_out + in_c * W_stride_in;
                    float *UG_out_base = UG_data + out_c * UG_stride_c;
                    for (int kh = 0; kh < p.K; kh++) {
                        int i = h_in - kh;
                        if (i >= 0 && i < p.H_out) {
                            float *UG_row_ptr = UG_out_base + i * UG_stride_h;
                            float *W_row_ptr = W_in_base + kh * W_stride_kh;
                            for (int kw = 0; kw < p.K; kw++) {
                                int j = w_in - kw;
                                if (j >= 0 && j < p.W_out) {
                                    sum += W_row_ptr[kw] * UG_row_ptr[j];
                                }
                            }
                        }
                    }
                }
                *dX_pad_row_ptr++ = sum;
            }
        }
    }

    Tensor *dX = tensor_unpad2d(dX_pad, p.padding);
    tensor_free(dX_pad);
    return dX;
}

// Unfolds padded input tensor X_pad into matrix X_col.
//
// X_col = (C_in * K * K) x (H_out * W_out) matrix
Tensor *im2col(Tensor *X_pad, int kernel_size, int stride) {

    // Local parameters
    int C_in = X_pad->shape[0];
    int X_pad_stride_in = X_pad->strides[0];
    int X_pad_stride_h = X_pad->strides[1];
    int H_padded = X_pad->shape[1];
    int W_padded = X_pad->shape[2];
    int H_out = (H_padded - kernel_size) / stride + 1;
    int W_out = (W_padded - kernel_size) / stride + 1;
    int X_col_rows = C_in * kernel_size * kernel_size;
    int X_col_cols = H_out * W_out;

    // Create output X_col tensor
    Tensor *X_col = tensor_create2d(X_col_rows, X_col_cols);

    // Reshape
    int col_idx = 0;
    for (int out_h = 0; out_h < H_out; out_h++) {
        for (int out_w = 0; out_w < W_out; out_w++) {
            int row_idx = 0;
            for (int c_in = 0; c_in < C_in; c_in++) {
                float *X_pad_in_base = X_pad->data + c_in * X_pad_stride_in;
                for (int kh = 0; kh < kernel_size; kh++) {
                    int h_in = out_h * stride + kh;
                    float *X_pad_row_ptr = X_pad_in_base + h_in * X_pad_stride_h;
                    for (int kw = 0; kw < kernel_size; kw++) {
                        int w_in = out_w * stride + kw;
                        X_col->data[row_idx * X_col_cols + col_idx] = X_pad_row_ptr[w_in];
                        row_idx++;
                    }
                }
            }
            col_idx++;
        }
    }
    return X_col;
}

Tensor *conv_layer_forward_im2col(ConvLayer *layer, const Tensor *input) {
    ConvParams p = conv_params_create(layer, input);

    // 1. Pad and transform input
    Tensor *X_pad = tensor_pad2d(input, p.padding);
    Tensor *X_col = im2col(X_pad, p.K, p.stride);

    // 2. Get view of weights in (C_out, C_in * K * K)
    Tensor *W_row = tensor_view(layer->weights, 2, (int[]){p.C_out, p.C_in * p.K * p.K});

    // 3. GEMM: Y_flat = W_row * X_col
    Tensor *Y_flat = tensor_create2d(p.C_out, p.H_out * p.W_out);
    tensor_matmul(Y_flat, W_row, X_col);

    // 4. Add biases (broadcast across spatial dimensions)
    for (int c = 0; c < p.C_out; c++) {
        float bias = layer->biases->data[c];
        for (int i = 0; i < p.H_out * p.W_out; i++) {
            Y_flat->data[c * (p.H_out * p.W_out) + i] += bias;
        }
    }

    // 5. Reshape to (C_out, H_out, W_out) - keeps ownership
    Tensor *Y = tensor_reshape_inplace(Y_flat, 3, (int[]){p.C_out, p.H_out, p.W_out});

    // 6. Cleanup temporary views
    tensor_free(W_row);
    tensor_free(X_pad);

    // 7. Cache for backward pass
    if (layer->input != NULL) {
        tensor_free(layer->input);
    }
    if (layer->output != NULL) {
        tensor_free(layer->output);
    }
    layer->input = X_col;
    layer->output = Y;

    return Y;
}

Tensor *col2im(Tensor *dX_col, int input_channels, int H_padded, int W_padded, int kernel_size,
               int stride) {
}

Tensor *conv_layer_backward_im2col(ConvLayer *layer, const Tensor *upstream_grad) {
}
