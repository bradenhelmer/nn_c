/*
 * gpu_layer_ops.cu - GPU layer passes implementations.
 *
 */
#include "gpu/gpu_tensor.h"
#include "gpu_layer_ops.h"
#include <float.h>
#include <math.h>

#define THREADS 256
#define BLOCKS(size) ((size) + (THREADS) - 1) / (THREADS)

// Linear Layer
__global__ void linear_add_bias_kernel(float *Y, const float *b, const int batch_size,
                                       const int out_features) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = batch_size * out_features;
    if (idx < total) {
        const int feature_idx = idx % out_features;
        Y[idx] += b[feature_idx];
    }
}

static void _gpu_linear_add_bias(const GPUTensor *Y, const GPUTensor *b, const int batch_size,
                                 const int out_features) {
    const int total = batch_size * out_features;
    const int blocks = BLOCKS(total);
    linear_add_bias_kernel<<<blocks, THREADS>>>(Y->d_data, b->d_data, batch_size, out_features);
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

    _gpu_linear_add_bias(Y, b, batch_size, out_features);
    return Y;
}

__global__ void sum_columns_kernel(float *db, const float *dY, int batch_size, int out_features,
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

static void _gpu_sum_columns(float *db, const float *dY, int batch_size, int out_features,
                             float beta) {
    const int blocks = BLOCKS(out_features);
    sum_columns_kernel<<<blocks, THREADS>>>(db, dY, batch_size, out_features, beta);
}

GPUTensor *gpu_linear_layer_backward(cublasHandle_t cublas, GPUTensor *dY, const GPUTensor *X,
                                     const GPUTensor *dX, const GPUTensor *W, const GPUTensor *dW,
                                     const GPUTensor *db) {
    const int batch_size = X->shape[0];
    const int out_features = dW->shape[0];
    const int in_features = dW->shape[1];

    const float alpha = 1.0f;
    const float beta_accum = 1.0f; // For accumulating into dW, db
    const float beta_zero = 1.0f;

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
    _gpu_sum_columns(db->d_data, dY->d_data, batch_size, out_features, beta_accum);

    // 3. Input gradient: dX = dY @ W
    cublasSgemm_v2(cublas, CUBLAS_OP_N, CUBLAS_OP_N, in_features, batch_size, out_features, &alpha,
                   W->d_data, in_features, dY->d_data, out_features, &beta_zero, dX->d_data,
                   in_features);

    return (GPUTensor *)dX;
}

// Activation Layer
GPUTensor *gpu_activation_layer_forward(GPUTensor *output, GPUTensor *input, ActivationType activation_type) {
    switch (activation_type) {
    case ACTIVATION_SIGMOID:
    case ACTIVATION_RELU:
    case ACTIVATION_TANH:
    case ACTIVATION_LINEAR:
        break;
    }
    return NULL;
}

GPUTensor *gpu_activation_layer_backward(GPUTensor *upstream_grad, ActivationType activation_type) {
    return NULL;
}

// #define FMAX(a, b) (a) > (b) ? (a) : (b)
//
// __global__ void softmax_cross_entropy_loss_kernel(float *d_losses, const float *prediction,
//                                                   const float *target, const int batch_size,
//                                                   const int num_classes) {
//     extern __shared__ float shared_max[];
//     extern __shared__ float shared_sum[];
//     extern __shared__ float softmax[];
//
//     int global_i = threadIdx.x;
//
//     float val = FLT_MIN;
//     if (global_i < batch_size) {
//         val = prediction[global_i];
//     }
//     shared_max[global_i] = val;
//     __syncthreads();
//
//     // Parallel reduction to find max value
//     int stride = blockDim.x / 2;
//     while (stride > 0) {
//         if (global_i < stride) {
//             shared_max[global_i] = FMAX(shared_max[global_i], shared_max[global_i + stride]);
//         }
//         __syncthreads();
//         stride /= 2;
//     }
//
//     float block_max = shared_max[0];
//
//     float exp_val = 0.0f;
//     if (global_i < batch_size) {
//         exp_val = expf(val - block_max);
//     }
//     shared_sum[global_i] = val;
//     __syncthreads();
//
//     // Parallel reduction for sum similar to reduction for max
//     stride = blockDim.x / 2;
//     while (stride > 0) {
//         if (global_i > stride) {
//             shared_sum[global_i] += shared_sum[global_i + stride];
//         }
//         __syncthreads();
//         stride /= 2;
//     }
//
//     // Normalize by sum
//     float block_sum = shared_sum[0];
//     if (global_i < batch_size) {
//         softmax[global_i] = exp_val / block_sum;
//     }
//
//     // Do cross entropy.
//
// }
//
// // Loss functions
// void gpu_softmax_cross_entropy_loss(float *d_losses, const GPUTensor *prediction,
//                                     const GPUTensor *target, const int batch_size,
//                                     const int num_classes) {
// }
