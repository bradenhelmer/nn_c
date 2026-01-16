/*
 * gpu_tensor.h -  GPU tensor declarations
 *
 * NOTE: GPUTensor struct is currently fully exposed, which breaks encapsulation
 * and exposes CUDA implementation details. Clients should not directly access
 * d_data or other fields - use provided functions instead.
 *
 * Future versions may make GPUTensor opaque to allow implementation changes
 * (e.g., switching to different GPU backends) without breaking client code.
 */

#ifndef GPU_TENSOR_H
#define GPU_TENSOR_H

#include "core/tensor.h"
#include <cuda_runtime.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define GPU_MAX_RANK 4

typedef struct {

    // DEVICE MEMORY
    float *d_data;   // Device pointer to GPU memory
    size_t capacity; // Allocated bytes ( >= size * 4)

    // METADATA (lives on host, passed to kernels by value)
    int ndim;
    int shape[GPU_MAX_RANK];   // Fixed max rank for simplicity
    int strides[GPU_MAX_RANK]; // Precomputed strides.
    int size;                  // Total elements

    // MEMORY MANAGEMENT
    int owns_data; // If 1, gpu_tensor_free() will cudaFree(d_data). If 0, borrows d_data pointer
    int device_id;
} GPUTensor;

// ============================================================================
// Lifecycle - Ownership rules
// ============================================================================

// Allocates new GPU memory and returns a GPUTensor that owns it
// Caller must free with gpu_tensor_free() when done
// Sets owns_data=1
GPUTensor *gpu_tensor_create(int ndim, int shape[GPU_MAX_RANK]);

// Allocates new GPU memory matching the shape of 'other'
// Caller must free with gpu_tensor_free() when done
// Sets owns_data=1
GPUTensor *gpu_tensor_create_like(const GPUTensor *other);

// Frees the GPUTensor struct and (if owns_data=1) the GPU memory
// Safe to call with NULL
void gpu_tensor_free(GPUTensor *gpu_t);

// ============================================================================
// Host <-> Device Transfers
// ============================================================================

// Allocates GPU memory, copies from CPU tensor, returns owning GPUTensor
// Caller must free returned tensor with gpu_tensor_free()
// Sets owns_data=1
GPUTensor *gpu_tensor_from_cpu(Tensor *cpu_t);

// Allocates CPU tensor, copies from GPU, returns owning CPU Tensor
// Caller must free returned tensor with tensor_free()
Tensor *gpu_tensor_to_cpu(GPUTensor *gpu_t);

// Copies data from host memory to existing GPU tensor (overwrites gpu_t->d_data)
// Does NOT take ownership of host_ptr - caller retains responsibility
void gpu_tensor_copy_from_host(GPUTensor *gpu_t, float *host_ptr, size_t num_elements);

void gpu_tensor_copy_from_host_async(GPUTensor *gpu_t, float *host_ptr, size_t num_elements,
                                     cudaStream_t stream);

// Copies data from GPU tensor to host memory (overwrites host_ptr)
// Does NOT take ownership of host_ptr - caller retains responsibility
void gpu_tensor_copy_to_host(float *host_ptr, GPUTensor *gpu_t, size_t num_elements);

void gpu_tensor_copy_to_host_async(float *host_ptr, GPUTensor *gpu_t, size_t num_elements,
                                   cudaStream_t stream);

// ============================================================================
// Shape Manipulation
// ============================================================================

// Returns a NEW GPUTensor with different shape but same data pointer
// Returned tensor does NOT own memory (owns_data=0) - caller must not free it independently
// Original tensor retains ownership
GPUTensor *gpu_tensor_reshape(GPUTensor *gpu_t, int ndim, int new_shape[GPU_MAX_RANK]);

// Returns a NEW GPUTensor viewing same data with different shape
// Returned tensor does NOT own memory (owns_data=0) - caller must not free it independently
// Original tensor retains ownership
GPUTensor *gpu_tensor_view(const GPUTensor *gpu_t, int ndim, int new_shape[GPU_MAX_RANK]);

// Returns a NEW GPUTensor with 1D shape viewing same data
// Returned tensor does NOT own memory (owns_data=0) - caller must not free it independently
// Original tensor retains ownership
GPUTensor *gpu_tensor_flatten(GPUTensor *gpu_t);

// ============================================================================
// Convolution Utilities
// ============================================================================
GPUTensor *gpu_tensor_pad2d(const GPUTensor *gpu_t, int padding);
GPUTensor *gpu_tensor_unpad2d(const GPUTensor *gpu_t, int padding);

int gpu_tensor_is_contiguous(GPUTensor *gpu_t);

// Device <-> device operations
void gpu_tensor_copy(GPUTensor *dest, const GPUTensor *src);
void gpu_tensor_copy_async(GPUTensor *dest, const GPUTensor *src, cudaStream_t stream);

// Utils
void gpu_tensor_print_shape(const GPUTensor *gpu_t);

// Operations
void gpu_tensor_scale(GPUTensor *dest, const GPUTensor *src, float scale);

#ifdef __cplusplus
}
#endif

#endif /* ifndef GPU_TENSOR_H */
