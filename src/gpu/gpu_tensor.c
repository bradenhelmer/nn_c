/*
 * gpu_tensor.c -  GPU tensor implementations.
 */
#include "gpu_tensor.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

static inline void _gpu_tensor_set_size_metadata(GPUTensor *gpu_t, int ndim,
                                                 int shape[GPU_MAX_RANK]) {

    gpu_t->ndim = ndim;

    // Only copy ndim elements to avoid reading past the end of dynamically allocated CPU tensors
    memcpy(gpu_t->shape, shape, sizeof(int) * ndim);
    // Zero-initialize remaining elements
    for (int i = ndim; i < GPU_MAX_RANK; i++) {
        gpu_t->shape[i] = 0;
    }

    // Compute strides left to right
    int strides_local[GPU_MAX_RANK] = {0};
    strides_local[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides_local[i] = strides_local[i + 1] * shape[i + 1];
    }
    memcpy(gpu_t->strides, strides_local, sizeof(int) * ndim);
    // Zero-initialize remaining elements
    for (int i = ndim; i < GPU_MAX_RANK; i++) {
        gpu_t->strides[i] = 0;
    }
    gpu_t->size = strides_local[0] * shape[0];
    gpu_t->capacity = sizeof(float) * gpu_t->size;
}

GPUTensor *gpu_tensor_create(int ndim, int shape[GPU_MAX_RANK]) {
    GPUTensor *gpu_t = (GPUTensor *)malloc(sizeof(GPUTensor));
    _gpu_tensor_set_size_metadata(gpu_t, ndim, shape);
    cudaMalloc((void **)&gpu_t->d_data, sizeof(float) * gpu_t->capacity);
    cudaMemset((void *)gpu_t->d_data, 0, gpu_t->capacity);
    gpu_t->owns_data = 1;
    return gpu_t;
}

GPUTensor *gpu_tensor_create_like(GPUTensor *other) {
    return gpu_tensor_create(other->ndim, other->shape);
}

void gpu_tensor_free(GPUTensor *gpu_t) {
    if (gpu_t == NULL) {
        return;
    }
    if (gpu_t->owns_data) {
        cudaFree(gpu_t->d_data);
    }
    free(gpu_t);
}

// Host <-> device tranfers
GPUTensor *gpu_tensor_from_cpu(Tensor *cpu_t) {
    assert(cpu_t->ndim <= 4);
    int shape[GPU_MAX_RANK] = {0};
    for (int i = 0; i < cpu_t->ndim; i++) {
        shape[i] = cpu_t->shape[i];
    }
    GPUTensor *gpu_t = gpu_tensor_create(cpu_t->ndim, cpu_t->shape);
    gpu_tensor_copy_from_host(gpu_t, cpu_t->data, cpu_t->size);
    return gpu_t;
}

Tensor *gpu_tensor_to_cpu(GPUTensor *gpu_t) {
    Tensor *cpu_t = tensor_create(gpu_t->ndim, gpu_t->shape);
    gpu_tensor_copy_to_host(cpu_t->data, gpu_t, gpu_t->size);
    return cpu_t;
}

void gpu_tensor_copy_from_host(GPUTensor *gpu_t, float *host_ptr, size_t num_elements) {
    cudaMemcpy((void *)gpu_t->d_data, (const void *)host_ptr, sizeof(float) * num_elements,
               cudaMemcpyHostToDevice);
}

void gpu_tensor_copy_from_host_async(GPUTensor *gpu_t, float *host_ptr, size_t num_elements,
                                     cudaStream_t stream) {
    cudaMemcpyAsync((void *)gpu_t->d_data, (const void *)host_ptr, sizeof(float) * num_elements,
                    cudaMemcpyHostToDevice, stream);
}

void gpu_tensor_copy_to_host(float *host_ptr, GPUTensor *gpu_t, size_t num_elements) {
    cudaMemcpy((void *)host_ptr, (const void *)gpu_t->d_data, sizeof(float) * num_elements,
               cudaMemcpyDeviceToHost);
}

void gpu_tensor_copy_to_host_async(float *host_ptr, GPUTensor *gpu_t, size_t num_elements,
                                   cudaStream_t stream) {
    cudaMemcpyAsync((void *)host_ptr, (const void *)gpu_t->d_data, sizeof(float) * num_elements,
                    cudaMemcpyDeviceToHost, stream);
}

static int _check_gpu_new_size(__attribute__((unused)) const GPUTensor *t, int ndim,
                               int *new_shape) {
    assert(ndim <= 4);
    int new_size = 1;
    for (int i = 0; i < ndim; i++) {
        new_size *= new_shape[i];
    }
    assert(new_size == t->size);
    return new_size;
}

// Shape manipulation
GPUTensor *gpu_tensor_reshape(GPUTensor *gpu_t, int ndim, int new_shape[GPU_MAX_RANK]) {

    const int new_size = _check_gpu_new_size(gpu_t, ndim, new_shape);

    gpu_t->ndim = ndim;
    memcpy(gpu_t->shape, new_shape, ndim * sizeof(int));

    // Compute strides left to right
    int strides_local[GPU_MAX_RANK];
    strides_local[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        strides_local[i] = strides_local[i + 1] * new_shape[i + 1];
    }
    memcpy(gpu_t->strides, strides_local, sizeof(int) * GPU_MAX_RANK);

    gpu_t->size = new_size;
    return gpu_t;
}

GPUTensor *gpu_tensor_view(GPUTensor *gpu_t, int ndim, int new_shape[GPU_MAX_RANK]) {
    _check_gpu_new_size(gpu_t, ndim, new_shape);
    GPUTensor *view = (GPUTensor *)malloc(sizeof(GPUTensor));
    _gpu_tensor_set_size_metadata(view, ndim, new_shape);
    view->d_data = gpu_t->d_data;
    view->owns_data = 0;
    return view;
}

GPUTensor *gpu_tensor_flatten(GPUTensor *gpu_t) {
    GPUTensor *view = (GPUTensor *)malloc(sizeof(GPUTensor));
    view->d_data = gpu_t->d_data;
    view->ndim = 1;

    view->shape[0] = gpu_t->size;
    view->shape[1] = 0;
    view->shape[2] = 0;
    view->shape[3] = 0;

    view->strides[0] = 1;
    view->strides[1] = 0;
    view->strides[2] = 0;
    view->strides[3] = 0;
    view->size = gpu_t->size;
    view->capacity = gpu_t->capacity;

    view->owns_data = 0;

    return view;
}

int gpu_tensor_is_contiguous(GPUTensor *gpu_t) {
    if (gpu_t->ndim == 0) {
        return 1;
    }

    int expected_stride = 1;
    for (int i = gpu_t->ndim - 1; i >= 0; i--) {
        if (gpu_t->shape[i] > 1) {
            if (gpu_t->strides[i] != expected_stride) {
                return 0;
            }
            expected_stride *= gpu_t->shape[i];
        }
    }
    return 1;
}

// Device <-> device opeations
// void gpu_tensor_copy(float *dest, float *src);
// void gpu_tensor_async(float *dest, float *src, cudaStream_t stream);
