/*
 * gpu_activations.cu - GPU activation function implementations.
 *
 */
#include "gpu/gpu_activations.h"
#include <assert.h>

#define THREADS 256
#define BLOCKS(size) ((size) + (THREADS) - (1)) / (THREADS)

static __device__ float _relu_scalar(float x) {
    return x <= 0.0f ? 1.0f : 0.0f;
}

__global__ void relu_kernel(float *output, float *input, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = _relu_scalar(input[idx]);
    }
}

void gpu_tensor_relu(GPUTensor *output, GPUTensor *input) {
    assert(output->size == input->size);
    const int size = output->size;
    relu_kernel<<<BLOCKS(size), THREADS>>>(output->d_data, input->d_data, size);
}

static __device__ float _relu_scalar_derivative(float relu_output) {
    return relu_output > 0.f ? 1.f : 0.f;
}

__global__ void relu_derivative_kernel(float *output, float *relu_output, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = _relu_scalar_derivative(relu_output[idx]);
    }
}

void gpu_tensor_relu_derivative(GPUTensor *output, GPUTensor *relu_output) {
    assert(output->size == relu_output->size);
    const int size = output->size;
    relu_derivative_kernel<<<BLOCKS(size), THREADS>>>(output->d_data, relu_output->d_data, size);
}

static __device__ float _sigmoid_scalar(float x) {
    return 1.f / (1.f + expf(-(x)));
}

__global__ void sigmoid_kernel(float *output, float *input, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = _sigmoid_scalar(input[idx]);
    }
}

void gpu_tensor_sigmoid(GPUTensor *output, GPUTensor *input) {
    assert(output->size == input->size);
    const int size = output->size;
    sigmoid_kernel<<<BLOCKS(size), THREADS>>>(output->d_data, input->d_data, size);
}

static __device__ float _sigmoid_scalar_derivative(float sigmoid_output) {
    return sigmoid_output * (1.0f - sigmoid_output);
}

__global__ void sigmoid_derivative_kernel(float *output, float *sigmoid_output, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = _sigmoid_scalar_derivative(sigmoid_output[idx]);
    }
}

void gpu_tensor_sigmoid_derivative(GPUTensor *output, GPUTensor *sigmoid_output) {
    assert(output->size == sigmoid_output->size);
    const int size = output->size;
    sigmoid_derivative_kernel<<<BLOCKS(size), THREADS>>>(output->d_data, sigmoid_output->d_data,
                                                         size);
}

static __device__ float _tanh_scalar(float x) {
    return tanhf(x);
}

__global__ void tanh_kernel(float *output, float *input, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = _tanh_scalar(input[idx]);
    }
}

void gpu_tensor_tanh(GPUTensor *output, GPUTensor *input) {
    assert(output->size == input->size);
    const int size = output->size;
    tanh_kernel<<<BLOCKS(size), THREADS>>>(output->d_data, input->d_data, size);
}

static __device__ float _tanh_scalar_derivative(float tanh_output) {
    return 1.0f - (tanh_output * tanh_output);
}

__global__ void tanh_derivative_kernel(float *output, float *tanh_output, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = _tanh_scalar_derivative(tanh_output[idx]);
    }
}

void gpu_tensor_tanh_derivative(GPUTensor *output, GPUTensor *tanh_output) {
    assert(output->size == tanh_output->size);
    const int size = output->size;
    tanh_derivative_kernel<<<BLOCKS(size), THREADS>>>(output->d_data, tanh_output->d_data, size);
}

void gpu_tensor_linear(GPUTensor *output, GPUTensor *input) {
    assert(output->size == input->size);
    cudaMemcpy(output->d_data, input->d_data, sizeof(float) * input->size,
               cudaMemcpyDeviceToDevice);
}

__global__ void linear_derivative_kernel(float *output, const int size) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = 1.0f;
    }
}
void gpu_tensor_linear_derivative(GPUTensor *output, GPUTensor *linear_output) {
    assert(output->size == linear_output->size);
    const int size = output->size;
    linear_derivative_kernel<<<BLOCKS(size), THREADS>>>(output->d_data, size);
}
