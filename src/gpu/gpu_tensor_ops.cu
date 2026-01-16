/*
 * gpu_tensor_ops.cu - Operations & kernels for GPU Tensors
 */
#include "gpu_tensor.h"
#include <assert.h>

#define THREADS 256
#define BLOCKS(size) ((size) + (THREADS) - (1)) / (THREADS)

__global__ void _tensor_scale_kernel(float *dest, float *src, float scale, int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dest[idx] = src[idx] * scale;
    }
}

void gpu_tensor_scale(GPUTensor *dest, const GPUTensor *src, float scale) {
    assert(dest->size == src->size);
    const int size = dest->size;
    _tensor_scale_kernel<<<BLOCKS(size), THREADS>>>(dest->d_data, src->d_data, scale, size);
}

__global__ void _tensor_pad2d_kernel(float *padded, const float *input, int batch_size, int C,
                                     int H_in, int W_in, int H_out, int W_out, int padding) {

    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= batch_size * C * H_out * W_out) {
        return;
    }

    // Convert linear index to 4D coordinates (batch, c, h_out, w_out)
    const int batch = idx / (C * H_out * W_out);
    const int remainder = idx % (C * H_out * W_out);
    const int c = remainder / (H_out * W_out);
    const int h_out = (remainder % (H_out * W_out)) / W_out;
    const int w_out = remainder % W_out;

    // Check if (h_out, w_out) in padded region
    if (h_out < padding || h_out >= H_in + padding || w_out < padding || w_out >= W_in + padding) {
        padded[idx] = 0.0f;
    } else {
        const int h_in = h_out - padding;
        const int w_in = w_out - padding;
        const int input_idx = batch * (C * H_in * W_in) + c * (H_in * W_in) + h_in * W_in + w_in;
        padded[idx] = input[input_idx];
    }
}

GPUTensor *gpu_tensor_pad2d(const GPUTensor *gpu_t, int padding) {
    const int batch_size = gpu_t->shape[0];
    const int C = gpu_t->shape[1];
    const int H_in = gpu_t->shape[2];
    const int W_in = gpu_t->shape[3];

    const int H_out = H_in + 2 * padding;
    const int W_out = W_in + 2 * padding;

    int padded_shape[GPU_MAX_RANK] = {batch_size, C, H_out, W_out};
    GPUTensor *padded = gpu_tensor_create(4, padded_shape);

    const int total_elements = batch_size * C * H_out * W_out;
    _tensor_pad2d_kernel<<<BLOCKS(total_elements), THREADS>>>(
        padded->d_data, gpu_t->d_data, batch_size, C, H_in, W_in, H_out, W_out, padding);

    return padded;
}

__global__ void _tensor_unpad2d_kernel(float *unpadded, const float *input, int batch_size, int C,
                                       int H_in, int W_in, int H_out, int W_out, int padding) {
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= batch_size * C * H_out * W_out) {
        return;
    }

    // Convert linear index to 4D coordinates (batch, c, h_out, w_out)
    const int batch = idx / (C * H_out * W_out);
    const int remainder = idx % (C * H_out * W_out);
    const int c = remainder / (H_out * W_out);
    const int h_out = (remainder % (H_out * W_out)) / W_out;
    const int w_out = remainder % W_out;
    const int h_in = h_out + padding;
    const int w_in = w_out + padding;
    const int input_idx = batch * (C * H_in * W_in) + c * (H_in * W_in) + h_in * W_in + w_in;
    unpadded[idx] = input[input_idx];
}

GPUTensor *gpu_tensor_unpad2d(const GPUTensor *gpu_t, int padding) {
    const int batch_size = gpu_t->shape[0];
    const int C = gpu_t->shape[1];
    const int H_in = gpu_t->shape[2];
    const int W_in = gpu_t->shape[3];

    const int H_out = H_in - 2 * padding;
    const int W_out = W_in - 2 * padding;

    int unpadded_shape[GPU_MAX_RANK] = {batch_size, C, H_out, W_out};
    GPUTensor *unpadded = gpu_tensor_create(4, unpadded_shape);
    const int total_elements = batch_size * C * H_out * W_out;
    _tensor_unpad2d_kernel<<<BLOCKS(total_elements), THREADS>>>(
        unpadded->d_data, gpu_t->d_data, batch_size, C, H_in, W_in, H_out, W_out, padding);

    return unpadded;
}
