/*
 * gpu_layer_ops.cu - GPU layer passes implementations.
 *
 */
#include "gpu/gpu_activations.h"
#include "gpu/gpu_tensor.h"
#include "gpu_layer_ops.h"
#include "layers/layer.h"
#include "layers/layer_internal.h"
#include <float.h>
#include <math.h>
#include <stdio.h>

#define THREADS 256
#define BLOCKS(size) ((size) + (THREADS) - 1) / (THREADS)

// ==============================================================================
// GPU LINEAR LAYER
// ==============================================================================

__global__ void _linear_add_bias_kernel(float *Y, const float *b, const int batch_size,
                                        const int out_features) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * out_features;
    if (idx < total) {
        const int feature_idx = idx % out_features;
        Y[idx] += b[feature_idx];
    }
}

GPUTensor *gpu_linear_layer_forward(cublasHandle_t cublas, GPUTensor *Y, const GPUTensor *X,
                                    const GPUTensor *W, const GPUTensor *b) {
    const int batch_size = X->shape[0];
    const int in_features = X->shape[1];
    const int out_features = W->shape[0];

    const float alpha = 1.0f;
    const float beta = 0.0f;

    cublasSgemm_v2(cublas, CUBLAS_OP_T, CUBLAS_OP_N, out_features, batch_size, in_features, &alpha,
                   W->d_data, in_features, X->d_data, in_features, &beta, Y->d_data, out_features);

    _linear_add_bias_kernel<<<BLOCKS(batch_size * out_features), THREADS>>>(
        Y->d_data, b->d_data, batch_size, out_features);
    return Y;
}

__global__ void _sum_columns_kernel(float *db, const float *dY, int batch_size, int out_features,
                                    float beta) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j < out_features) {
        float sum = 0.0f;
        for (int i = 0; i < batch_size; i++) {
            sum += dY[i * out_features + j];
        }

        if (beta == 0.0f) {
            db[j] = sum;
        } else {
            db[j] += sum;
        }
    }
}

GPUTensor *gpu_linear_layer_backward(cublasHandle_t cublas, GPUTensor *dY, const GPUTensor *X,
                                     const GPUTensor *dX, const GPUTensor *W, const GPUTensor *dW,
                                     const GPUTensor *db) {
    const int batch_size = X->shape[0];
    const int out_features = dW->shape[0];
    const int in_features = dW->shape[1];

    const float alpha = 1.0f;
    const float beta_accum = 1.0f; // For accumulating into dW, db
    const float beta_zero = 0.0f;

    // 1. Compute outer product of upstream gradient and input.
    //
    // dW^T (col-major) = X^T @ dY
    //
    // cuBLAS sees X as X^T, dY as dY^T
    // op(A) = X^T (no transpose), op(B) = dY (transpose dY^T)
    cublasSgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_T, in_features, out_features, batch_size, &alpha,
                   X->d_data, in_features, dY->d_data, out_features, &beta_accum, dW->d_data,
                   in_features);

    // 2. Bias gradient: db + sum(dY, axis=0)
    _sum_columns_kernel<<<BLOCKS(out_features), THREADS>>>(db->d_data, dY->d_data, batch_size,
                                                           out_features, beta_accum);

    // 3. Input gradient: dX = dY @ W
    cublasSgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N, in_features, batch_size, out_features, &alpha,
                   W->d_data, in_features, dY->d_data, out_features, &beta_zero, dX->d_data,
                   in_features);

    return (GPUTensor *)dX;
}

// ==============================================================================
// GPU ACTIVATION LAYER
// ==============================================================================

GPUTensor *gpu_activation_layer_forward(GPUTensor *output, GPUTensor *input,
                                        ActivationType activation_type) {
    switch (activation_type) {
    case ACTIVATION_SIGMOID:
        gpu_tensor_sigmoid(output, input);
        break;
    case ACTIVATION_RELU:
        gpu_tensor_relu(output, input);
        break;
    case ACTIVATION_TANH:
        gpu_tensor_tanh(output, input);
        break;
    case ACTIVATION_LINEAR:
        gpu_tensor_linear(output, input);
        break;
    }
    return output;
}

GPUTensor *gpu_activation_layer_backward(GPUTensor *output, GPUTensor *upstream_grad,
                                         ActivationType activation_type) {
    switch (activation_type) {
    case ACTIVATION_SIGMOID:
        gpu_tensor_sigmoid_derivative(output, upstream_grad);
        break;
    case ACTIVATION_RELU:
        gpu_tensor_relu_derivative(output, upstream_grad);
        break;
    case ACTIVATION_TANH:
        gpu_tensor_tanh_derivative(output, upstream_grad);
        break;
    case ACTIVATION_LINEAR:
        break;
    }
    return output;
}

// ==============================================================================
// GPU CONV2D LAYER PARAMS
// ==============================================================================

Conv2DParams gpu_conv2d_params_create(const Conv2DLayer *layer, const GPUTensor *input) {
    Conv2DParams p;
    p.C_in = layer->in_channels;
    p.C_out = layer->out_channels;
    p.K = layer->kernel_size;
    p.stride = layer->stride;
    p.padding = layer->padding;

    p.batch_size = input->shape[0];
    p.H_in = input->shape[2];
    p.W_in = input->shape[3];
    p.H_padded = p.H_in + 2 * p.padding;
    p.W_padded = p.W_in + 2 * p.padding;
    p.H_out = (p.H_padded - p.K) / p.stride + 1;
    p.W_out = (p.W_padded - p.K) / p.stride + 1;

    return p;
}

Conv2DParams gpu_conv2d_params_from_padded(const Conv2DLayer *layer,
                                           const GPUTensor *padded_input) {
    Conv2DParams p;
    p.C_in = layer->in_channels;
    p.C_out = layer->out_channels;
    p.K = layer->kernel_size;
    p.stride = layer->stride;
    p.padding = layer->padding;

    p.batch_size = padded_input->shape[0];
    p.H_padded = padded_input->shape[2];
    p.W_padded = padded_input->shape[3];
    p.H_in = p.H_padded - 2 * p.padding;
    p.W_in = p.W_padded - 2 * p.padding;
    p.H_out = (p.H_padded - p.K) / p.stride + 1;
    p.W_out = (p.W_padded - p.K) / p.stride + 1;

    return p;
}

// Create ConvParams directly from raw parameters - most flexible
Conv2DParams gpu_conv2d_params_make(const Conv2DLayer *layer, int batch_size, int H_in, int W_in) {
    Conv2DParams p;
    p.batch_size = batch_size;
    p.C_in = layer->in_channels;
    p.C_out = layer->out_channels;
    p.K = layer->kernel_size;
    p.stride = layer->stride;
    p.padding = layer->padding;
    p.H_in = H_in;
    p.W_in = W_in;
    p.H_padded = H_in + 2 * p.padding;
    p.W_padded = W_in + 2 * p.padding;
    p.H_out = (p.H_padded - p.K) / p.stride + 1;
    p.W_out = (p.W_padded - p.K) / p.stride + 1;
    return p;
}

// Create ConvParams from layer and upstream gradient shape (for backward pass)
Conv2DParams gpu_conv2d_params_from_upstream(const Conv2DLayer *layer,
                                             const GPUTensor *upstream_grad) {
    // upstream_grad shape: (C_out, H_out, W_out)
    const int batch_size = upstream_grad->shape[0];
    const int H_out = upstream_grad->shape[2];
    const int W_out = upstream_grad->shape[3];

    // Reverse the output dimension formula: H_out = (H_padded - K) / stride + 1
    const int H_padded = (H_out - 1) * layer->stride + layer->kernel_size;
    const int W_padded = (W_out - 1) * layer->stride + layer->kernel_size;
    const int H_in = H_padded - 2 * layer->padding;
    const int W_in = W_padded - 2 * layer->padding;

    return gpu_conv2d_params_make(layer, batch_size, H_in, W_in);
}

// ==============================================================================
// GPU CONV2D LAYER
// ==============================================================================

__global__ void _conv2d_layer_im2col_kernel(float *X_col, float *X_pad, int batch_size, int C_in,
                                            int H_out, int W_out, int K, int stride,
                                            int X_pad_stride_batch, int X_pad_stride_channel,
                                            int X_pad_stride_h) {

    // Calculate global index and do bounds check on output dimensions.
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int size = (C_in * K * K) * (H_out * W_out * batch_size);
    if (idx >= size) {
        return;
    }

    // Convert linear idx to 2D output (row, col)
    const int row = idx / (H_out * W_out * batch_size);
    const int col = idx % (H_out * W_out * batch_size);

    // Decode row -> (channel, kernel_row, kernel_col)
    const int c = row / (K * K);
    const int kh = (row % (K * K)) / K;
    const int kw = row % K;

    // Decode col -> (batch_idx, out_h, out_w)
    const int batch_idx = col / (H_out * W_out);
    const int spatial_col = col % (H_out * W_out);
    const int out_h = spatial_col / W_out;
    const int out_w = spatial_col % W_out;

    const int h_pad = out_h * stride + kh;
    const int w_pad = out_w * stride + kw;

    // Output: write to linearized position in X_col
    // Input: read from 4D X_pad (batch, channel, height, width)
    X_col[idx] = X_pad[batch_idx * X_pad_stride_batch + c * X_pad_stride_channel +
                       h_pad * X_pad_stride_h + w_pad];
}

static GPUTensor *_gpu_conv2d_layer_im2col(Conv2DLayer *layer, GPUTensor *X_pad) {
    Conv2DParams p = gpu_conv2d_params_from_padded(layer, X_pad);
    const int batch_size = p.batch_size;
    const int C_in = p.C_in;
    const int H_out = p.H_out;
    const int W_out = p.W_out;
    const int K = p.K;
    const int stride = p.stride;
    // X_pad shape: (batch, C, H_pad, W_pad)
    const int X_pad_stride_batch = X_pad->strides[0];
    const int X_pad_stride_channel = X_pad->strides[1];
    const int X_pad_stride_h = X_pad->strides[2];
    const int X_col_rows = C_in * K * K;
    const int X_col_cols = H_out * W_out * batch_size;

    int X_col_shape[GPU_MAX_RANK] = {X_col_rows, X_col_cols};
    GPUTensor *X_col = gpu_tensor_create(2, X_col_shape);

    _conv2d_layer_im2col_kernel<<<BLOCKS(X_col->size), THREADS>>>(
        X_col->d_data, X_pad->d_data, batch_size, C_in, H_out, W_out, K, stride, X_pad_stride_batch,
        X_pad_stride_channel, X_pad_stride_h);
    return X_col;
}

__global__ void _gpu_conv2d_layer_add_bias_kernel(float *Y_flat, float *bias, int C_out,
                                                  int Y_flat_cols) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= C_out * Y_flat_cols) {
        return;
    }
    const int c = idx / Y_flat_cols;
    Y_flat[idx] += bias[c];
}

// Transpose kernel: Converts Y_flat (C_out, H_out*W_out*batch) to Y (batch, C_out, H_out, W_out)
//
// Memory layout transformation:
//   Y_flat[c, col] where col encodes (batch_idx * H_out*W_out + h*W_out + w)
//   → Y[batch_idx, c, h, w]
//
// Strategy: Each thread handles one OUTPUT element in Y
// 1. Calculate which output element: decode idx to (batch_idx, c, h, w)
// 2. Calculate where to READ from Y_flat:
//    - row = c
//    - col = batch_idx * (H_out*W_out) + h*W_out + w
//    - Y_flat_index = c * Y_flat_stride + col
// 3. Write: Y[idx] = Y_flat[Y_flat_index]
__global__ void _conv2d_transpose_output_kernel(float *Y, const float *Y_flat, int batch_size,
                                                int C_out, int H_out, int W_out) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= batch_size * C_out * H_out * W_out) {
        return;
    }
    const int batch_idx = idx / (C_out * H_out * W_out);
    const int remainder = idx % (C_out * H_out * W_out);
    const int c = remainder / (H_out * W_out);
    const int h_out = (remainder % (H_out * W_out)) / W_out;
    const int w_out = remainder % W_out;

    const int col = batch_idx * (H_out * W_out) + h_out * W_out + w_out;
    const int Y_flat_index = c * (H_out * W_out * batch_size) + col;

    Y[idx] = Y_flat[Y_flat_index];
}

GPUTensor *gpu_conv2d_layer_forward(cublasHandle_t cublas, Conv2DLayer *layer,
                                    GPUTensor **input_cache_ptr, GPUTensor *Y, const GPUTensor *X,
                                    const GPUTensor *W, const GPUTensor *b) {
    Conv2DParams p = gpu_conv2d_params_create(layer, X);
    const int batch_size = p.batch_size;
    const int C_in = p.C_in;
    const int C_out = p.C_out;
    const int H_out = p.H_out;
    const int W_out = p.W_out;
    const int K = p.K;

    const int Y_flat_cols = H_out * W_out * batch_size;

    GPUTensor *X_pad = gpu_tensor_pad2d(X, p.padding);
    GPUTensor *X_col = _gpu_conv2d_layer_im2col(layer, X_pad);

    int W_row_shape[GPU_MAX_RANK] = {C_out, C_in * K * K, 0, 0};
    GPUTensor *W_row = gpu_tensor_view(W, 2, W_row_shape);

    int Y_flat_shape[GPU_MAX_RANK] = {C_out, Y_flat_cols, 0, 0};
    GPUTensor *Y_flat = gpu_tensor_create(2, Y_flat_shape);

    // 1. CuBLAS GEMM: Y_flat = W_row × X_col
    // In cuBLAS (column-major), this becomes: Y_flat^T = X_col^T × W_row^T
    const float alpha = 1.0f;
    const float beta = 0.0f;
    cublasSgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N, Y_flat_cols, C_out, C_in * K * K, &alpha,
                   X_col->d_data, Y_flat_cols, W_row->d_data, C_in * K * K, &beta, Y_flat->d_data,
                   Y_flat_cols);

    // 2. ADD BIASES
    _gpu_conv2d_layer_add_bias_kernel<<<BLOCKS(Y_flat->size), THREADS>>>(Y_flat->d_data, b->d_data,
                                                                         C_out, Y_flat_cols);

    // 3. Transpose Y_flat to Y's 4D shape
    // Y_flat: (C_out, H_out*W_out*batch) → Y: (batch, C_out, H_out, W_out)
    _conv2d_transpose_output_kernel<<<BLOCKS(Y->size), THREADS>>>(Y->d_data, Y_flat->d_data,
                                                                  batch_size, C_out, H_out, W_out);

    gpu_tensor_free(W_row);
    gpu_tensor_free(X_pad);
    gpu_tensor_free(Y_flat);

    // Store X_col in the provided cache location (for backward pass)
    // X_col must be stored instead of just being freed, but input_cache_ptr
    // may already have the wrong shape (from being allocated to match the 4D input).
    // Solution: free the old cache and transfer ownership of X_col.
    if (input_cache_ptr != NULL) {
        // Free the old cache (which was allocated with wrong shape for conv2d)
        if (*input_cache_ptr != NULL) {
            gpu_tensor_free(*input_cache_ptr);
        }
        // Replace it with X_col directly
        *input_cache_ptr = X_col; // Transfer ownership
    } else {
        // No cache provided, so free X_col
        gpu_tensor_free(X_col);
    }

    return Y;
}

// Reshapes upstream gradient dY in to flattened form:
// (batch_size, C_out, H_out, W_out) -> (C_out, H_out * W_out * batch_size)
__global__ void _reshape_upstream_kernel(float *dY_flat, float *dY, int batch_size, int C_out,
                                         int H_out, int W_out) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= C_out * H_out * W_out * batch_size) {
        return;
    }

    // Retrieve output indices
    const int row = idx / (H_out * W_out * batch_size);
    const int col = idx % (H_out * W_out * batch_size);

    // Retrieve input indices
    const int batch_idx = col / (H_out * W_out);
    const int spatial_idx = col % (H_out * W_out);
    const int h = spatial_idx / W_out;
    const int w = spatial_idx % W_out;

    int src_idx = batch_idx * (C_out * H_out * W_out) + row * (H_out * W_out) + h * W_out + w;
    dY_flat[idx] = dY[src_idx];
}

__global__ void _gpu_conv2d_layer_sum_bias_gradient_kernel(float *db, float *dY_flat, int C_out,
                                                           int dY_flat_cols) {
    // Get src index.
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (idx >= C_out * dY_flat_cols) {
        return;
    }

    // Gest destination idx
    const int c = idx / dY_flat_cols;
    atomicAdd(&db[c], dY_flat[idx]);
}

// col2im: Scatter dX_col back to dX_pad, accumulating overlapping contributions
//
// dX_col layout: (C_in * K * K, H_out * W_out * batch_size)
// - Each column is one output spatial position from one batch
// - Each row is one (channel, kernel_row, kernel_col) combination
//
// dX_pad layout: (batch_size, C_in, H_padded, W_padded)
//
// Strategy: Each thread handles one element of dX_col and adds it to the
// corresponding position in dX_pad. Multiple threads may write to the same
// dX_pad location (overlapping patches), so we use atomicAdd.
//
// Note: We WILL write gradients to padded regions here! That's expected -
// the padding participated in the forward pass, so it gets gradients.
// After col2im, we unpad to extract only the real input gradients.
__global__ void _conv2d_layer_col2im_kernel(float *dX_pad, float *dX_col, int batch_size, int C_in,
                                            int H_out, int W_out, int K, int stride,
                                            int dX_pad_stride_batch, int dX_pad_stride_in,
                                            int dX_pad_stride_h, int dX_col_row_stride,
                                            int dX_col_size) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    const int total = (C_in * K * K) * (H_out * W_out * batch_size);
    if (idx >= total) {
        return;
    }

    // Decode linear index to (row, col) in dX_col
    const int row = idx / (H_out * W_out * batch_size);
    const int col = idx % (H_out * W_out * batch_size);

    // Decode row -> (channel, kernel_row, kernel_col)
    const int c = row / (K * K);
    const int kh = (row % (K * K)) / K;
    const int kw = row % K;

    // Decode col -> (batch_idx, out_h, out_w)
    const int batch_idx = col / (H_out * W_out);
    const int spatial_col = col % (H_out * W_out);
    const int out_h = spatial_col / W_out;
    const int out_w = spatial_col % W_out;

    // Calculate position in padded input where this gradient contributes
    const int h_pad = out_h * stride + kh;
    const int w_pad = out_w * stride + kw;

    // Calculate linear index in dX_pad and accumulate gradient
    // (overlapping patches mean multiple threads may write to same location)
    const int dX_pad_idx =
        batch_idx * dX_pad_stride_batch + c * dX_pad_stride_in + h_pad * dX_pad_stride_h + w_pad;

    atomicAdd(&dX_pad[dX_pad_idx], dX_col[idx]);
}

static GPUTensor *_gpu_conv2d_layer_col2im(GPUTensor *dX_col, Conv2DParams *p) {
    const int C_in = p->C_in;
    const int H_out = p->H_out;
    const int W_out = p->W_out;
    const int H_padded = p->H_padded;
    const int W_padded = p->W_padded;
    const int K = p->K;
    const int stride = p->stride;
    const int batch_size = p->batch_size;
    const int dX_col_row_stride = dX_col->strides[0];

    int dX_pad_shape[GPU_MAX_RANK] = {batch_size, C_in, H_padded, W_padded};
    GPUTensor *dX_pad = gpu_tensor_create(4, dX_pad_shape);

    const int dX_col_size = dX_col->size;
    const int dX_pad_stride_batch = dX_pad->strides[0];
    const int dX_pad_stride_in = dX_pad->strides[1];
    const int dX_pad_stride_h = dX_pad->strides[2];

    _conv2d_layer_col2im_kernel<<<BLOCKS(dX_col_size), THREADS>>>(
        dX_pad->d_data, dX_col->d_data, batch_size, C_in, H_out, W_out, K, stride,
        dX_pad_stride_batch, dX_pad_stride_in, dX_pad_stride_h, dX_col_row_stride, dX_col_size);

    return dX_pad;
}

GPUTensor *gpu_conv2d_layer_backward(cublasHandle_t cublas, Conv2DLayer *layer, GPUTensor *dY,
                                     const GPUTensor *X_col, const GPUTensor *dX,
                                     const GPUTensor *W, const GPUTensor *dW, const GPUTensor *db) {
    Conv2DParams p = gpu_conv2d_params_from_upstream(layer, dY);
    const int batch_size = p.batch_size;
    const int C_in = p.C_in;
    const int C_out = p.C_out;
    const int H_out = p.H_out;
    const int W_out = p.W_out;
    const int K = p.K;

    // 1. Reshape upstream grad to (C_out, H_out * W_out * batch_size)
    int dY_flat_shape[GPU_MAX_RANK] = {C_out, H_out * W_out * batch_size, 0, 0};
    GPUTensor *dY_flat = gpu_tensor_create(2, dY_flat_shape);
    _reshape_upstream_kernel<<<BLOCKS(dY_flat->size), THREADS>>>(dY_flat->d_data, dY->d_data,
                                                                 batch_size, C_out, H_out, W_out);

    // 2. Sum bias gradient over spatial dimensions
    _gpu_conv2d_layer_sum_bias_gradient_kernel<<<BLOCKS(dY_flat->size), THREADS>>>(
        db->d_data, dY_flat->d_data, C_out, dY_flat_shape[1]);

    // 3. Weight gradient: dY_flat x X_col ^ T
    //
    // dY_flat: (C_out, H_out * W_out * batch_size)
    // X_col : (C_in * K * K, H_out * W_out * batch_size)
    // grad_W_flat: (C_out, C_in * K * K)
    //
    // grad_W_flat = dY_flat @ (X_col)^T
    //
    // CUBLAS:
    // (grad_W_flat)^T = (dY_flat @ (X_col)^T)^T
    // = X_col @ (dY_flat)^T
    //
    // CUBLAS sees X_col^T transpose to get X_col
    // CUBLAS sees dY_flat^T, do not transpose to achieve implicit
    //
    // OP_T(X_col^T) = X_col = A: (C_in * K * K, H_out * W_out * batch_size)
    //  M = C_in * K * K
    //  K = H_out * W_out * batch_size
    //  lda = H_out * W_out * batch_size
    //
    // OP_N(dY_flat^T) = dY_flat^T = B: (H_out * W_out * batch_size, C_out)
    //  N = C_out
    //  ldb = H_out * W_out * batch_size
    //
    //  ldc = C_in * K * K
    int grad_W_flat_shape[GPU_MAX_RANK] = {C_out, C_in * K * K};
    GPUTensor *grad_W_flat = gpu_tensor_create(2, grad_W_flat_shape);

    const float alpha = 1.0f;
    const float beta_accum = 1.0f; // For accumulating into dW, db
    const float beta_zero = 0.0f;

    cublasSgemm_v2(cublas, CUBLAS_OP_T, CUBLAS_OP_N, X_col->shape[0], C_out, X_col->shape[1],
                   &alpha, X_col->d_data, X_col->shape[1], dY_flat->d_data, dY_flat->shape[1],
                   &beta_accum, grad_W_flat->d_data, grad_W_flat->shape[1]);

    // 4. Input gradient: W_row^T x dY_flat, col2im
    //
    // W_row = view(W, {C_out, C_in * K * K}): (C_out, C_in * K * K)
    // dY_flat: (C_out, H_out * W_out * batch_size)
    // dX_col: (C_in * K * K, H_out * W_out * batch_size)
    //
    // dX_col = (W_row)^T * dY_flat
    //
    // CUBLAS:
    // (dX_col)^T = ((W_row)^T * dY_flat)^T
    // = (dY_flat)^T * W_row
    //
    // CUBLAS sees (dY_flat)^T, do not transpose to achieve implicit
    // CUBLAS sees (W_row)^T, transpose to get W_row
    //
    // OP_N(dY_flat^T) = dY_flat^T = A: (H_out * W_out * batch_size, C_out)
    // M = H_out * W_out * batch_size
    // K = C_out
    // lda = H_out * W_out * batch_size
    //
    // OP_T(W_row^T) = W_row = B: (C_out, C_in * K * K)
    // N = C_in * K * K
    // ldb =  C_in * K * K
    //
    // ldc = H_out * W_out * batch_size
    int dX_col_shape[GPU_MAX_RANK] = {C_in * K * K, H_out * W_out * batch_size, 0, 0};
    GPUTensor *dX_col = gpu_tensor_create(2, dX_col_shape);

    int W_row_shape[GPU_MAX_RANK] = {C_out, C_in * K * K, 0, 0};
    GPUTensor *W_row = gpu_tensor_view(W, 2, W_row_shape);

    cublasSgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_T, dX_col->shape[1], dX_col->shape[0], C_out,
                   &alpha, dY_flat->d_data, dY_flat->shape[1], W_row->d_data, W_row->shape[1],
                   &beta_zero, dX_col->d_data, dX_col->shape[1]);

    // 5. col2im: convert dX_col back to padded spatial format
    GPUTensor *dX_pad = _gpu_conv2d_layer_col2im(dX_col, &p);

    // 6. Remove padding to get final input gradient
    // dX_pad: (batch, C_in, H_padded, W_padded) -> dX: (batch, C_in, H_in, W_in)
    GPUTensor *dX_unpadded = gpu_tensor_unpad2d(dX_pad, p.padding);
    gpu_tensor_copy((GPUTensor *)dX, dX_unpadded);

    // Clean up temporary tensors
    gpu_tensor_free(dY_flat);
    gpu_tensor_free(grad_W_flat);
    gpu_tensor_free(dX_col);
    gpu_tensor_free(W_row);
    gpu_tensor_free(dX_pad);
    gpu_tensor_free(dX_unpadded);

    return (GPUTensor *)dX;
}

// ==============================================================================
// GPU MAX POOLING LAYER
// ==============================================================================

__global__ void _maxpool_forward_kernel(float *Y, const float *X, int *max_indices, int batch_size,
                                        int C, int H_in, int W_in, int H_out, int W_out, int stride,
                                        int pool_size) {
    // Each thread handles one output pool position
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pools = batch_size * C * H_out * W_out;
    if (idx >= total_pools) {
        return;
    }

    // Decode linear index to 4D coordinates: (batch, channel, h_out, w_out)
    const int batch_idx = idx / (C * H_out * W_out);
    const int remainder = idx % (C * H_out * W_out);
    const int c = remainder / (H_out * W_out);
    const int h_out = (remainder % (H_out * W_out)) / W_out;
    const int w_out = remainder % W_out;

    // Calculate starting position of pool window in input
    const int h_start = h_out * stride;
    const int w_start = w_out * stride;

    // Find maximum value within the pool window
    float max_val = -INFINITY;
    int max_offset = 0;

    for (int m = 0; m < pool_size; m++) {
        for (int n = 0; n < pool_size; n++) {
            // Index into input using INPUT dimensions (H_in, W_in)
            const int X_idx = batch_idx * (C * H_in * W_in) + c * (H_in * W_in) +
                              (h_start + m) * W_in + (w_start + n);
            const float val = X[X_idx];
            if (val > max_val) {
                max_val = val;
                max_offset = m * pool_size + n; // Store relative offset within pool
            }
        }
    }

    // Write output and store which input element was selected
    Y[idx] = max_val;
    max_indices[idx] = max_offset;
}

GPUTensor *gpu_maxpool_layer_forward(MaxPoolLayer *layer, GPUTensor *Y, const GPUTensor *X,
                                     int *max_indices) {
    const int batch_size = X->shape[0];
    const int total_pools = batch_size * layer->input_c * layer->output_h * layer->output_w;

    _maxpool_forward_kernel<<<BLOCKS(total_pools), THREADS>>>(
        Y->d_data, X->d_data, max_indices, batch_size, layer->input_c, layer->input_h,
        layer->input_w, layer->output_h, layer->output_w, layer->stride, layer->pool_size);

    return Y;
}

__global__ void _maxpool_backward_kernel(float *dX, const float *dY, const int *max_indices,
                                         int batch_size, int C, int H_in, int W_in, int H_out,
                                         int W_out, int stride, int pool_size) {
    // Each thread handles one upstream gradient (one output pool position)
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total_pools = batch_size * C * H_out * W_out;
    if (idx >= total_pools) {
        return;
    }

    // Decode linear index to 4D coordinates: (batch, channel, h_out, w_out)
    const int batch_idx = idx / (C * H_out * W_out);
    const int remainder = idx % (C * H_out * W_out);
    const int c = remainder / (H_out * W_out);
    const int h_out = (remainder % (H_out * W_out)) / W_out;
    const int w_out = remainder % W_out;

    // 1. Read the upstream gradient for this pool
    const float grad = dY[idx];

    // 2. Look up which input element was the max (stored as relative offset)
    const int max_offset = max_indices[idx];
    const int m = max_offset / pool_size; // row within pool
    const int n = max_offset % pool_size; // col within pool

    // 3. Convert to absolute input coordinates
    const int h_in = h_out * stride + m;
    const int w_in = w_out * stride + n;

    // 4. Scatter gradient to the input position that was selected as max
    const int dX_idx = batch_idx * (C * H_in * W_in) + c * (H_in * W_in) + h_in * W_in + w_in;
    atomicAdd(&dX[dX_idx], grad);
}

GPUTensor *gpu_maxpool_layer_backward(MaxPoolLayer *layer, GPUTensor *dY, GPUTensor *dX,
                                      int *max_indices) {
    const int batch_size = dY->shape[0];

    const int total_pools = batch_size * layer->input_c * layer->output_h * layer->output_w;

    _maxpool_backward_kernel<<<BLOCKS(total_pools), THREADS>>>(
        dX->d_data, dY->d_data, max_indices, batch_size, layer->input_c, layer->input_h,
        layer->input_w, layer->output_h, layer->output_w, layer->stride, layer->pool_size);

    return dX;
}

// ==============================================================================
// GPU FLATTEN LAYER
// ==============================================================================

GPUTensor *gpu_flatten_layer_forward(FlattenLayer *layer, GPUTensor *Y, const GPUTensor *X) {
    // Flatten just reshapes from (batch, C, H, W) to (batch, C*H*W)
    // No computation needed - data is already contiguous
    // Just copy the data pointer reference
    gpu_tensor_copy(Y, X);
    return Y;
}

GPUTensor *gpu_flatten_layer_backward(FlattenLayer *layer, GPUTensor *dY, GPUTensor *dX) {
    // Backward pass just reshapes from (batch, features) back to (batch, C, H, W)
    // No computation needed - just copy the gradient data
    gpu_tensor_copy(dX, dY);
    return dX;
}

// ==============================================================================
// GPU RESHAPE LAYER
// ==============================================================================

GPUTensor *gpu_reshape_layer_forward(ReshapeLayer *layer, GPUTensor *Y, const GPUTensor *X) {
    // Reshape just changes the view of the data without moving it
    // Data is already contiguous, so just copy to the pre-allocated output shape
    gpu_tensor_copy(Y, X);
    return Y;
}

GPUTensor *gpu_reshape_layer_backward(ReshapeLayer *layer, GPUTensor *dY, GPUTensor *dX) {
    // Backward pass reshapes gradient back to original input shape
    // No computation needed - just copy the gradient data to the pre-allocated shape
    gpu_tensor_copy(dX, dY);
    return dX;
}
