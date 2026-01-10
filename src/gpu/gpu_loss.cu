/*
 * gpu_loss.cu - GPU loss function implementations.
 */
#include "gpu_loss.h"
#include <cfloat>
#include <cmath>
#include <cuda_runtime.h>

#define THREADS 256
#define EPSILON 1e-7f

#define FULL_WARP_MASK 0xFFFFFFFF
#define WARP_OFFSET_START 16
#define WARP_SIZE 32

// ==============================================================================
// REDUCTION UTILITIES
// ==============================================================================

// Warp-level reduction primitives (CUDA warp = 32 threads that execute in lockstep)
__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = WARP_OFFSET_START; offset > 0; offset >>= 1) {
        val = fmaxf(val, __shfl_down_sync(FULL_WARP_MASK, val, offset));
    }
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = WARP_OFFSET_START; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(FULL_WARP_MASK, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
__device__ float block_reduce_max(float val) {
    __shared__ float warp_results[WARP_SIZE];

    val = warp_reduce_max(val);

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0) {
        warp_results[warp_id] = val;
    }

    __syncthreads();

    int num_warps = blockDim.x / WARP_SIZE;
    if (warp_id == 0) {
        val = (lane_id < num_warps) ? warp_results[lane_id] : -FLT_MAX;
        val = warp_reduce_max(val);
    }
    return val;
}

__device__ float block_reduce_sum(float val) {
    __shared__ float warp_results[WARP_SIZE];

    val = warp_reduce_sum(val);

    int warp_id = threadIdx.x / WARP_SIZE;
    int lane_id = threadIdx.x % WARP_SIZE;

    if (lane_id == 0) {
        warp_results[warp_id] = val;
    }

    __syncthreads();

    int num_warps = blockDim.x / WARP_SIZE;
    if (warp_id == 0) {
        val = (lane_id < num_warps) ? warp_results[lane_id] : 0.0f;
        val = warp_reduce_sum(val);
    }
    return val;
}

// ==============================================================================
// SOFTMAX CROSS-ENTROPY FORWARD
// ==============================================================================

// Kernel: Compute softmax cross-entropy loss per sample
// Grid: batch_size blocks, each block handles one sample
// Block: 256 threads cooperating to process num_classes values
__global__ void softmax_cross_entropy_forward_kernel(float *d_per_sample_losses,
                                                     const float *d_logits, const float *d_target,
                                                     int batch_size, int num_classes) {
    int sample_idx = blockIdx.x;
    if (sample_idx >= batch_size) {
        return;
    }

    const float *logits = d_logits + sample_idx * num_classes;
    const float *target = d_target + sample_idx * num_classes;

    // Step 1: Find max logit for numerical stability
    // - Each thread processes multiple elements (stride loop: i = tid; i < num_classes; i +=
    // blockDim.x)
    // - Track max_val across elements this thread sees
    // - Call block_reduce_max() to get global max
    // - Broadcast result to all threads via shared memory
    int tid = threadIdx.x;
    float local_max = -FLT_MAX;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        local_max = fmaxf(local_max, logits[i]);
    }

    local_max = block_reduce_max(local_max);
    __shared__ float shared_max;
    if (tid == 0) {
        shared_max = local_max;
    }
    __syncthreads();

    // Step 2: Compute sum of exp(logits - max)
    // - Similar stride loop
    // - sum_exp += expf(logits[i] - max_val)
    // - Call block_reduce_sum()
    // - Broadcast via shared memory
    float local_sum = 0.0f;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        local_sum += __expf(logits[i] - shared_max);
    }

    local_sum = block_reduce_sum(local_sum);
    __shared__ float shared_sum;
    if (tid == 0) {
        shared_sum = local_sum;
    }
    __syncthreads();

    // Step 3: Compute cross-entropy loss
    // - Loop over elements computing: -target[i] * log(softmax[i])
    // - Where log(softmax[i]) = (logits[i] - max_val) - log(sum_exp + EPSILON)
    // - Reduce to get total loss for this sample
    float local_loss = 0.0f;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        local_loss += -target[i] * ((logits[i] - shared_max) - __logf(shared_sum + EPSILON));
    }
    local_loss = block_reduce_sum(local_loss);

    // Step 4: Thread 0 writes result
    if (tid == 0) {
        d_per_sample_losses[sample_idx] = local_loss;
    }
}

// Kernel: Reduce per-sample losses to batch average
__global__ void reduce_losses_kernel(float *d_total_loss, const float *d_per_sample_losses,
                                     int batch_size) {
    // 1. Each thread sums a subset of per-sample losses (stride loop)
    int tid = threadIdx.x;
    float local_sum = 0.0f;
    for (int i = tid; i < batch_size; i += blockDim.x) {
        local_sum += d_per_sample_losses[i];
    }

    // 2. Use block_reduce_sum() to combine within block
    local_sum = block_reduce_sum(local_sum);

    // 3. Thread 0 of each block uses atomicAdd(d_total_loss, sum) to accumulate
    if (tid == 0) {
        atomicAdd(d_total_loss, local_sum);
    }
}

// ==============================================================================
// SOFTMAX CROSS-ENTROPY BACKWARD
// ==============================================================================

// Kernel: Compute gradient = softmax(logits) - target
__global__ void softmax_cross_entropy_backward_kernel(float *d_grad, const float *d_logits,
                                                      const float *d_target, int batch_size,
                                                      int num_classes) {
    int sample_idx = blockIdx.x;
    if (sample_idx >= batch_size) {
        return;
    }

    int tid = threadIdx.x;
    const float *logits = d_logits + sample_idx * num_classes;
    const float *target = d_target + sample_idx * num_classes;
    float *grad = d_grad + sample_idx * num_classes;

    // Step 1 & 2: Same as forward (find max, compute sum_exp)
    // You'll need these to compute softmax values
    float local_max = -FLT_MAX;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        local_max = fmaxf(local_max, logits[i]);
    }

    local_max = block_reduce_max(local_max);
    __shared__ float shared_max;
    if (tid == 0) {
        shared_max = local_max;
    }
    __syncthreads();

    float local_sum = 0.0f;
    for (int i = tid; i < num_classes; i += blockDim.x) {
        local_sum += __expf(logits[i] - shared_max);
    }

    local_sum = block_reduce_sum(local_sum);
    __shared__ float shared_sum;
    if (tid == 0) {
        shared_sum = local_sum;
    }
    __syncthreads();

    // Step 3: Compute gradient
    // - Stride loop over elements
    // - softmax[i] = exp(logits[i] - max_val) / sum_exp
    // - grad[i] = softmax[i] - target[i]
    for (int i = tid; i < num_classes; i += blockDim.x) {
        grad[i] = (__expf(logits[i] - shared_max) / shared_sum) - target[i];
    }
}

// ==============================================================================
// HOST FUNCTIONS
// ==============================================================================

float gpu_softmax_cross_entropy_loss(const GPUTensor *logits, const GPUTensor *target) {
    int batch_size = logits->shape[0];
    int num_classes = logits->shape[1];

    // 1. Allocate per-sample losses: cudaMalloc(&d_per_sample_losses, batch_size * sizeof(float))
    float *d_per_sample_losses;
    cudaMalloc((void **)&d_per_sample_losses, batch_size * sizeof(float));

    // 2. Launch forward kernel: <<<batch_size blocks, 256 threads>>>
    softmax_cross_entropy_forward_kernel<<<batch_size, THREADS>>>(
        d_per_sample_losses, logits->d_data, target->d_data, batch_size, num_classes);

    // 3. Allocate total loss: cudaMalloc(&d_total_loss, sizeof(float))
    float *d_total_loss;
    cudaMalloc((void **)&d_total_loss, sizeof(float));

    // 4. Zero it: cudaMemset(d_total_loss, 0, sizeof(float))
    cudaMemset(d_total_loss, 0.0f, sizeof(float));

    // 5. Launch reduction kernel
    reduce_losses_kernel<<<1, THREADS>>>(d_total_loss, d_per_sample_losses, batch_size);

    // 6. Copy result to host: cudaMemcpy(&total_loss, d_total_loss, ...)
    float total_loss;
    cudaMemcpy(&total_loss, d_total_loss, sizeof(float), cudaMemcpyDeviceToHost);

    // 7. Cleanup: cudaFree both allocations
    cudaFree(d_per_sample_losses);
    cudaFree(d_total_loss);

    // 8. Return: total_loss / batch_size
    return total_loss / batch_size;
}

void gpu_softmax_cross_entropy_backward(GPUTensor *grad, const GPUTensor *logits,
                                        const GPUTensor *target) {
    int batch_size = logits->shape[0];
    int num_classes = logits->shape[1];

    // 1. Launch backward kernel: <<<batch_size blocks, 256 threads>>>
    // 2. Kernel writes directly to grad->d_data
    // 3. No reduction or copying needed!
    softmax_cross_entropy_backward_kernel<<<batch_size, THREADS>>>(
        grad->d_data, logits->d_data, target->d_data, batch_size, num_classes);
}
