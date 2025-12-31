/*
 * gpu_tensor.h -  GPU tensor declarations
 */

#ifndef GPU_TENSOR_H
#define GPU_TENSOR_H

#include "tensor/tensor.h"
#include <cuda_runtime.h>
#include <stddef.h>

#define GPU_MAX_RANK 4

typedef struct {

    // DEVICE MEMORY
    float *d_data;   // Device pointer
    size_t capacity; // Allocated bytes ( >= size * 4)

    // METADATA (lives on host, passed to kernels by value)
    int ndim;
    int shape[GPU_MAX_RANK];   // Fixed max rank for simplicity
    int strides[GPU_MAX_RANK]; // Precomputed strides.
    int size;                  // Total elements

    // MEMORY MANAGEMENT
    int owns_data;
    int device_id;
} GPUTensor;

// Lifecycle
GPUTensor *gpu_tensor_create(int ndim, int shape[GPU_MAX_RANK]);
GPUTensor *gpu_tensor_create_like(GPUTensor *other);
void gpu_tensor_free(GPUTensor *gpu_t);

// Host <-> device tranfers
GPUTensor *gpu_tensor_from_cpu(Tensor *cpu_t);
Tensor *gpu_tensor_to_cpu(GPUTensor *gpu_t);
void gpu_tensor_copy_from_host(GPUTensor *gpu_t, float *host_ptr, size_t num_elements);
void gpu_tensor_copy_from_host_async(GPUTensor *gpu_t, float *host_ptr, size_t num_elements,
                                     cudaStream_t stream);
void gpu_tensor_copy_to_host(float *host_ptr, GPUTensor *gpu_t, size_t num_elements);
void gpu_tensor_copy_to_host_async(float *host_ptr, GPUTensor *gpu_t, size_t num_elements,
                                   cudaStream_t stream);

// Shape manipulation
GPUTensor *gpu_tensor_reshape(GPUTensor *gpu_t, int ndim, int new_shape[GPU_MAX_RANK]);
GPUTensor *gpu_tensor_view(GPUTensor *gpu_t, int ndim, int new_shape[GPU_MAX_RANK]);
GPUTensor *gpu_tensor_flatten(GPUTensor *gpu_t);
int gpu_tensor_is_contiguous(GPUTensor *gpu_t);

// Device <-> device opeations
void gpu_tensor_copy(float *dest, float *src);
void gpu_tensor_async(float *dest, float *src, cudaStream_t stream);

#endif /* ifndef GPU_TENSOR_H */
