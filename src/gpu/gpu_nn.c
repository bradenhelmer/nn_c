/*
 * gpu_nn.c - GPUNeuralNet executor implementations.
 */
#include "gpu_nn.h"
#include "gpu_tensor.h"
#include "nn/layer.h"
#include <cuda_runtime.h>
#include <stdlib.h>

static size_t _compute_workspace_size(NeuralNet *cpu_nn, int batch_size) {
    return 1;
}

// Lifecyle
GPUNeuralNet *gpu_nn_create_from_cpu_nn(NeuralNet *cpu_nn, int batch_size) {
    GPUNeuralNet *gpu_nn = (GPUNeuralNet *)malloc(sizeof(GPUNeuralNet));
    gpu_nn->cpu_nn = cpu_nn;
    gpu_nn->num_layers = cpu_nn->num_layers;
    gpu_nn->batch_size = batch_size;

    // 1. Count parameters
    gpu_nn->layer_param_offset = (int *)malloc(gpu_nn->num_layers);
    int param_count = 0;
    for (int i = 0; i < gpu_nn->num_layers; i++) {
        LayerParameters lp = layer_get_parameters(cpu_nn->layers[i]);
        if (lp.num_pairs > 0) {
            gpu_nn->layer_param_offset[i] = param_count;
            param_count += lp.num_pairs * 2;
        } else {
            gpu_nn->layer_param_offset[i] = -1;
        }
        layer_parameters_free(&lp);
    }
    gpu_nn->num_params = param_count;

    // 2. Allocate and copy parameters
    gpu_nn->d_params = (GPUTensor **)malloc(sizeof(GPUTensor *) * param_count);
    int param_idx = 0;
    for (int i = 0; i < gpu_nn->num_layers; i++) {
        LayerParameters lp = layer_get_parameters(cpu_nn->layers[i]);
        for (int j = 0; j < lp.num_pairs; j++) {
            Tensor *param = lp.pairs[j].param;

            // Create GPU tensor from same shape.
            gpu_nn->d_params[param_idx] = gpu_tensor_create(param->ndim, param->shape);

            // Copy from CPU
            gpu_tensor_copy_from_host(gpu_nn->d_params[param_idx], param->data, param->size);

            // Create gradient tensor (zeroed)
            gpu_nn->d_grads[param_idx] = gpu_tensor_create_like(gpu_nn->d_params[param_idx]);

            param_idx++;
        }
        layer_parameters_free(&lp);
    }

    // 3. Allocate activation cache
    // Actual tensors are allocated during forward pass from workspace.
    gpu_nn->d_activations = (GPUTensor **)malloc(sizeof(GPUTensor *) * gpu_nn->num_layers);

    // 4. Allocate layer auxiliary data.
    gpu_nn->layer_aux = (void **)malloc(sizeof(void *) * gpu_nn->num_layers);
    for (int i = 0; i < gpu_nn->num_layers; i++) {
        Layer *layer = cpu_nn->layers[i];
        switch (layer->type) {
        case LAYER_MAX_POOL: {
            // MaxPoolLayer *mp = (MaxPoolLayer *)layer->layer;
            // Compute output size for this batch
            // For now, defer until forward pass knows input shape.
            gpu_nn->layer_aux[i] = NULL;
        }
        case LAYER_FLATTEN:
            // Store shape metadata (allocated later)
            gpu_nn->layer_aux[i] = NULL;

        default:
            gpu_nn->layer_aux[i] = NULL;
        }
    }

    // 5. Compute and allocate workspace
    gpu_nn->workspace_size = _compute_workspace_size(cpu_nn, batch_size);
    cudaMalloc((void **)&gpu_nn->d_workspace, gpu_nn->workspace_size);
    gpu_nn->workspace_offset = 0;

    // 6. Initialize CUDA resources
    cublasCreate_v2(&gpu_nn->cublas);
    cudaStreamCreate(&gpu_nn->compute_stream);
    cudaStreamCreate(&gpu_nn->transfer_stream);

    // 7. Initialize optimizer state (Adam)
    gpu_nn->d_m = (GPUTensor **)malloc(sizeof(GPUTensor *) * param_count);
    gpu_nn->d_v = (GPUTensor **)malloc(sizeof(GPUTensor *) * param_count);
    for (int i = 0; i < param_count; i++) {
        gpu_nn->d_m[i] = gpu_tensor_create_like(gpu_nn->d_params[i]);
        gpu_nn->d_v[i] = gpu_tensor_create_like(gpu_nn->d_params[i]);
    }
    gpu_nn->t = 0;
    gpu_nn->beta1 = 0.9f;
    gpu_nn->beta2 = 0.999f;
    gpu_nn->eps = 1e-8f;
    return gpu_nn;
}

void gpu_nn_to_cpu_nn(GPUNeuralNet *gpu_nn) {
    int param_idx = 0;
    for (int i = 0; i < gpu_nn->num_layers; i++) {
        LayerParameters lp = layer_get_parameters(gpu_nn->cpu_nn->layers[i]);
        for (int j = 0; j < lp.num_pairs; j++) {
            Tensor *cpu_param = lp.pairs[j].param;
            GPUTensor *gpu_param = gpu_nn->d_params[param_idx];
            gpu_tensor_copy_to_host(cpu_param->data, gpu_param, cpu_param->size);
            param_idx++;
        }
        layer_parameters_free(&lp);
    }
}

void gpu_nn_free(GPUNeuralNet *gpu_nn) {

    // Free parameter/optimizer tensors
    for (int i = 0; i < gpu_nn->num_params; i++) {
        gpu_tensor_free(gpu_nn->d_params[i]);
        gpu_tensor_free(gpu_nn->d_grads[i]);
        gpu_tensor_free(gpu_nn->d_m[i]);
        gpu_tensor_free(gpu_nn->d_v[i]);
    }

    // Free actual storage pointers
    free(gpu_nn->d_params);
    free(gpu_nn->d_grads);
    free(gpu_nn->d_m);
    free(gpu_nn->d_v);

    // Free layer param offset array
    free(gpu_nn->layer_param_offset);

    // Free activation tensors and layer auxiliary data if present.
    for (int i = 0; i < gpu_nn->num_layers; i++) {
        gpu_tensor_free(gpu_nn->d_activations[i]);
        void *layer_auxiliary_data = gpu_nn->layer_aux[i];
        if (layer_auxiliary_data != NULL) {
            free(layer_auxiliary_data);
        }
    }
    // Free storage pointers
    free(gpu_nn->d_activations);
    free(gpu_nn->layer_aux);

    // Free workspace device memory
    cudaFree(gpu_nn->d_workspace);

    // Cleanup cuda resources
    cublasDestroy_v2(gpu_nn->cublas);
    cudaStreamDestroy(gpu_nn->compute_stream);
    cudaStreamDestroy(gpu_nn->transfer_stream);
}

// Training
GPUTensor *gpu_nn_forward(GPUNeuralNet *gpu_nn, GPUTensor *input) {

    // Cache initial input
    gpu_nn->input = input;

    workspace_reset(gpu_nn);
    GPUTensor *current = input;

    for (int i = 0; i < gpu_nn->num_layers; i++) {
        Layer *layer = gpu_nn->cpu_nn->layers[i];
        switch (layer->type) {
        case LAYER_CONV_2D: {
            ConvLayer *conv_layer = (ConvLayer *)layer->layer;
            int p_idx = gpu_nn->layer_param_offset[i];
            GPUTensor *weights = gpu_nn->d_params[p_idx];
            GPUTensor *biases = gpu_nn->d_params[p_idx + 1];
            // current = conv_layer_forward_gpu(...);
        }
        case LAYER_LINEAR: {
            LinearLayer *linear_layer = (LinearLayer *)layer->layer;
            int p_idx = gpu_nn->layer_param_offset[i];
            GPUTensor *weights = gpu_nn->d_params[p_idx];
            GPUTensor *biases = gpu_nn->d_params[p_idx + 1];
            // current = linear_layer_forward_gpu(...);
        }
        case LAYER_ACTIVATION: {
            ActivationLayer *al = (ActivationLayer *)layer->layer;
            // current = activation_forward_gpu(...)
        }
        case LAYER_MAX_POOL: {
            MaxPoolLayer *mpl = (MaxPoolLayer *)layer->layer;
            // current = maxpool_forward_gpu(...)
        }
        case LAYER_FLATTEN: {
            // current = flatten_forward_gpu(...)
        }
        }
        gpu_nn->d_activations[i] = current;
    }
    return current;
}

void gpu_nn_backward(GPUNeuralNet *gpu_nn, GPUTensor *target) {
    GPUTensor *output = gpu_nn->d_activations[gpu_nn->num_layers - 1];
    GPUTensor *grad = workspace_alloc_tensor(gpu_nn, output->shape);

    // gpu_softmax_cross_entropy_backward(grad, output, target)

    for (int i = gpu_nn->num_layers; i >= 0; --i) {
        Layer *layer = gpu_nn->cpu_nn->layers[i];
        GPUTensor *layer_input = (i == 0) ? gpu_nn->input : gpu_nn->d_activations[i - 1];
        switch (layer->type) {
        case LAYER_CONV_2D: {
            ConvLayer *conv_layer = (ConvLayer *)layer->layer;
            int p_idx = gpu_nn->layer_param_offset[i];
            GPUTensor *weights = gpu_nn->d_params[p_idx];
            GPUTensor *grad_weights = gpu_nn->d_grads[p_idx];
            GPUTensor *grad_biases = gpu_nn->d_grads[p_idx + 1];
            // current = conv_layer_backward_gpu(...);
        }
        case LAYER_LINEAR: {
            LinearLayer *linear_layer = (LinearLayer *)layer->layer;
            int p_idx = gpu_nn->layer_param_offset[i];
            GPUTensor *weights = gpu_nn->d_params[p_idx];
            GPUTensor *grad_weights = gpu_nn->d_grads[p_idx];
            GPUTensor *grad_biases = gpu_nn->d_grads[p_idx + 1];
            // current = linear_layer_backward_gpu(...);
        }
        case LAYER_ACTIVATION: {
            ActivationLayer *al = (ActivationLayer *)layer->layer;
            // current = activation_backward_gpu(...)
        }
        case LAYER_MAX_POOL: {
            MaxPoolLayer *mpl = (MaxPoolLayer *)layer->layer;
            // current = maxpool_backward_gpu(...)
        }
        case LAYER_FLATTEN: {
            // current = flatten_backward_gpu(...)
        }
        }
    }
}

void gpu_nn_zero_gradients(GPUNeuralNet *gpu_nn) {
}
void gpu_nn_optimizer_step(GPUNeuralNet *gpu_nn, float learning_rate);

// Batch training for convenience
float gpu_nn_train_batch(GPUNeuralNet *gpu_nn, Tensor *host_input, Tensor *host_target,
                         int batch_size);

// Evaluation
void gpu_nn_predict(GPUNeuralNet *gpu_nn, Tensor *host_input, Tensor *host_output, int batch_size);

// Workspace functions
void workspace_reset(GPUNeuralNet *gpu_nn) {
}
float *workspace_alloc(GPUNeuralNet *gpu_nn, size_t bytes) {
}
GPUTensor *workspace_alloc_tensor(GPUNeuralNet *gpu_nn, int shape[GPU_MAX_RANK]) {
}
