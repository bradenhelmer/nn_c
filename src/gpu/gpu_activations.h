/*
 * gpu_activations.h - GPU activation function declarations.
 *
 */

#ifndef GPU_ACTIVATIONS_H
#define GPU_ACTIVATIONS_H
#include "gpu/gpu_tensor.h"

void gpu_tensor_relu(GPUTensor *output, const GPUTensor *input);
void gpu_tensor_relu_derivative(GPUTensor *output, const GPUTensor *relu_output);

void gpu_tensor_sigmoid(GPUTensor *output, const GPUTensor *input);
void gpu_tensor_sigmoid_derivative(GPUTensor *output, const GPUTensor *sigmoid_output);

void gpu_tensor_tanh(GPUTensor *output, const GPUTensor *input);
void gpu_tensor_tanh_derivative(GPUTensor *output, const GPUTensor *tanh_output);

void gpu_tensor_linear(GPUTensor *output, const GPUTensor *input);
void gpu_tensor_linear_derivative(GPUTensor *output, const GPUTensor *linear_output);

#endif /* ifndef GPU_ACTIVATIONS_H */
