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

void gpu_tensor_scale(const GPUTensor *dest, const GPUTensor *src, float scale) {
    assert(dest->size == src->size);
    const int size = dest->size;
    _tensor_scale_kernel<<<BLOCKS(size), THREADS>>>(dest->d_data, src->d_data, scale, size);
}
