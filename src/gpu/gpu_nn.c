/*
 * gpu_nn.c - GPUNeuralNet executor implementations.
 */
#include "gpu_nn.h"
#include "gpu_layer_ops.h"
#include "gpu_tensor.h"
#include "nn/layer.h"
#include <assert.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <string.h>

static size_t _compute_workspace_size(NeuralNet *cpu_nn, int batch_size, InputShape input_shape) {
    size_t total = 0;

    // Use dynamic input shape instead of hardcoded MNIST dimensions
    int h = input_shape.height;
    int w = input_shape.width;
    int c = input_shape.channels;

    for (int i = 0; i < cpu_nn->num_layers; i++) {
        Layer *layer = cpu_nn->layers[i];
        size_t activation_size = 0;

        switch (layer->type) {
        case LAYER_CONV_2D: {
            ConvLayer *cfg = (ConvLayer *)layer->layer;
            int h_out = (h + 2 * cfg->padding - cfg->kernel_size) / cfg->stride + 1;
            int w_out = (w + 2 * cfg->padding - cfg->kernel_size) / cfg->stride + 1;

            // Output activation
            activation_size = batch_size * cfg->out_channels * h_out * w_out * sizeof(float);

            // Im2col buffer
            int im2col_rows = cfg->in_channels * cfg->kernel_size * cfg->kernel_size;
            int im2col_cols = h_out * w_out * batch_size;
            size_t im2col_size = im2col_rows * im2col_cols * sizeof(float);
            total += im2col_size;

            h = h_out;
            w = w_out;
            c = cfg->out_channels;
            break;
        }
        case LAYER_MAX_POOL: {
            MaxPoolLayer *cfg = (MaxPoolLayer *)layer->layer;
            int h_out = (h - cfg->pool_size) / cfg->stride + 1;
            int w_out = (w - cfg->pool_size) / cfg->stride + 1;

            activation_size = batch_size * c * h_out * w_out * sizeof(float);

            h = h_out;
            w = w_out;
            break;
        }
        case LAYER_FLATTEN: {
            int flat_size = c * h * w;
            activation_size = batch_size * flat_size * sizeof(float);
            // Update dimensions for next layer
            c = flat_size;
            h = 1;
            w = 1;
            break;
        }
        case LAYER_LINEAR: {
            LinearLayer *cfg = (LinearLayer *)layer->layer;
            activation_size = batch_size * cfg->output_size * sizeof(float);
            c = cfg->output_size;
            break;
        }
        case LAYER_ACTIVATION: {
            // Same size as input
            activation_size = batch_size * c * h * w * sizeof(float);
            break;
        }
        }

        // Align each allocation
        total += (activation_size + 255) & ~((size_t)255);
    }

    total *= 2;

    // Add safety margin
    total += 1024 * 1024; // 1 MB buffer

    return total;
}

// Lifecyle
GPUNeuralNet *gpu_nn_create_from_cpu_nn(NeuralNet *cpu_nn, int batch_size, InputShape input_shape) {
    GPUNeuralNet *gpu_nn = (GPUNeuralNet *)malloc(sizeof(GPUNeuralNet));
    gpu_nn->cpu_nn = cpu_nn;
    gpu_nn->num_layers = cpu_nn->num_layers;
    gpu_nn->batch_size = batch_size;
    gpu_nn->learning_rate = cpu_nn->learning_rate;
    gpu_nn->input_shape = input_shape;

    // 1. Count parameters
    gpu_nn->layer_param_offset = (int *)malloc(sizeof(int) * gpu_nn->num_layers);
    int param_count = 0;
    for (int i = 0; i < gpu_nn->num_layers; i++) {
        LayerParameters lp = layer_get_parameters(cpu_nn->layers[i]);
        if (lp.num_pairs > 0) {
            gpu_nn->layer_param_offset[i] = param_count;
            param_count += lp.num_pairs;
        } else {
            gpu_nn->layer_param_offset[i] = -1;
        }
        layer_parameters_free(&lp);
    }
    gpu_nn->num_params = param_count;

    // 2. Allocate and copy parameters
    gpu_nn->d_params = (GPUTensor **)malloc(sizeof(GPUTensor *) * param_count);
    gpu_nn->d_grads = (GPUTensor **)malloc(sizeof(GPUTensor *) * param_count);
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
    for (int i = 0; i < gpu_nn->num_layers; i++) {
        gpu_nn->d_activations[i] = NULL;
    }

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

    // 5. Compute and allocate workspace using dynamic input shape
    gpu_nn->workspace_size = _compute_workspace_size(cpu_nn, batch_size, input_shape);
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
        GPUTensor *d_activation = gpu_nn->d_activations[i];
        if (d_activation != NULL) {
            gpu_tensor_free(d_activation);
        }
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
            // ConvLayer *conv_layer = (ConvLayer *)layer->layer;
            // int p_idx = gpu_nn->layer_param_offset[i];
            // GPUTensor *weights = gpu_nn->d_params[p_idx];
            // GPUTensor *biases = gpu_nn->d_params[p_idx + 1];
            // current = conv_layer_forward_gpu(...);
            break;
        }
        case LAYER_LINEAR: {
            LinearLayer *linear_layer = (LinearLayer *)layer->layer;
            int p_idx = gpu_nn->layer_param_offset[i];
            GPUTensor *weights = gpu_nn->d_params[p_idx];
            GPUTensor *biases = gpu_nn->d_params[p_idx + 1];
            current = gpu_linear_layer_forward(gpu_nn->cublas, current, current, weights, biases);
            break;
        }
        case LAYER_ACTIVATION: {
            // ActivationLayer *al = (ActivationLayer *)layer->layer;
            // // current = activation_forward_gpu(...)
            break;
        }
        case LAYER_MAX_POOL: {
            // MaxPoolLayer *mpl = (MaxPoolLayer *)layer->layer;
            // if (gpu_nn->layer_aux[i] == NULL) {
            //     int h_out = (current->shape[2] - mpl->pool_size) / mpl->stride + 1;
            //     int w_out = (current->shape[3] - mpl->pool_size) / mpl->stride + 1;
            //     int indices_size = gpu_nn->batch_size * current->shape[1] * h_out * w_out;
            //     cudaMalloc(&gpu_nn->layer_aux[i], indices_size * sizeof(int));
            // }
            // int *d_indices = (int *)gpu_nn->layer_aux[i];
            // // current = maxpool_forward_gpu(...)
            break;
        }
        case LAYER_FLATTEN: {
            // current = flatten_forward_gpu(...)
            break;
        }
        }
        gpu_nn->d_activations[i] = current;
    }
    return current;
}

void gpu_nn_backward(GPUNeuralNet *gpu_nn, GPUTensor *target) {
    GPUTensor *output = gpu_nn->d_activations[gpu_nn->num_layers - 1];
    GPUTensor *grad = workspace_alloc_tensor(gpu_nn, output->ndim, output->shape);

    // gpu_softmax_cross_entropy_backward(grad, output, target)

    for (int i = gpu_nn->num_layers - 1; i >= 0; --i) {
        Layer *layer = gpu_nn->cpu_nn->layers[i];
        GPUTensor *layer_input = (i == 0) ? gpu_nn->input : gpu_nn->d_activations[i - 1];
        switch (layer->type) {
        case LAYER_CONV_2D: {
            // ConvLayer *conv_layer = (ConvLayer *)layer->layer;
            // int p_idx = gpu_nn->layer_param_offset[i];
            // GPUTensor *weights = gpu_nn->d_params[p_idx];
            // GPUTensor *grad_weights = gpu_nn->d_grads[p_idx];
            // GPUTensor *grad_biases = gpu_nn->d_grads[p_idx + 1];
            // current = conv_layer_backward_gpu(...);
            break;
        }
        case LAYER_LINEAR: {
            int p_idx = gpu_nn->layer_param_offset[i];
            GPUTensor *weights = gpu_nn->d_params[p_idx];
            GPUTensor *grad_weights = gpu_nn->d_grads[p_idx];
            GPUTensor *grad_biases = gpu_nn->d_grads[p_idx + 1];
            grad = gpu_linear_layer_backward(gpu_nn->cublas, grad, layer_input, weights,
                                             grad_weights, grad_biases);
            break;
        }
        case LAYER_ACTIVATION: {
            // ActivationLayer *al = (ActivationLayer *)layer->layer;
            // current = activation_backward_gpu(...)
            break;
        }
        case LAYER_MAX_POOL: {
            // MaxPoolLayer *mpl = (MaxPoolLayer *)layer->layer;
            // current = maxpool_backward_gpu(...)
            break;
        }
        case LAYER_FLATTEN: {
            // current = flatten_backward_gpu(...)
            break;
        }
        }
    }
}

void gpu_nn_zero_gradients(GPUNeuralNet *gpu_nn) {
    for (int i = 0; i < gpu_nn->num_params; i++) {
        cudaMemset(gpu_nn->d_grads[i]->d_data, 0, gpu_nn->d_grads[i]->size * sizeof(float));
    }
}

void gpu_nn_scale_gradients(GPUNeuralNet *gpu_nn, float scale) {
}

void gpu_nn_optimizer_step(GPUNeuralNet *gpu_nn) {
}

float gpu_nn_compute_loss(GPUNeuralNet *gpu_nn, GPUTensor *prediction, GPUTensor *target) {
    const int batch_size = prediction->shape[0];
    const int num_classes = prediction->shape[1];

    // Allocate temporary for per-sample losses
    float *d_losses = workspace_alloc(gpu_nn, batch_size * sizeof(float));

    // Launch softmax cross entropy loss kernel
    gpu_softmax_cross_entropy_loss(d_losses, prediction, target, batch_size, num_classes);
}

// Evaluation
void gpu_nn_predict(GPUNeuralNet *gpu_nn, Tensor *host_input, Tensor *host_output, int batch_size) {
}
float gpu_nn_evaluate_accuracy(GPUNeuralNet *gpu_nn, Dataset *val_data, GPUTensor *d_input,
                               float *h_input_pinned) {
}

// Workspace functions
void workspace_reset(GPUNeuralNet *gpu_nn) {
    gpu_nn->workspace_offset = 0;
}

float *workspace_alloc(GPUNeuralNet *gpu_nn, size_t bytes) {
    // Align to 256 bytes for coalesced access
    size_t aligned = (bytes + 255) & ~((size_t)255);

    assert(gpu_nn->workspace_offset + aligned <= gpu_nn->workspace_size);

    float *ptr = (float *)((char *)gpu_nn->d_workspace + gpu_nn->workspace_offset);
    gpu_nn->workspace_offset += aligned;

    return ptr;
}

GPUTensor *workspace_alloc_tensor(GPUNeuralNet *gpu_nn, int ndim, int shape[GPU_MAX_RANK]) {
    int size = 1;
    for (int i = 0; i < ndim; i++) {
        size *= shape[i];
    }

    // Allocate from workspace.
    float *d_data = workspace_alloc(gpu_nn, sizeof(float) * size);

    // Create host GPUTensor wrapper
    GPUTensor *gpu_t = (GPUTensor *)malloc(sizeof(GPUTensor));
    gpu_t->d_data = d_data;
    gpu_t->ndim = ndim;
    memcpy(gpu_t->shape, shape, sizeof(int) * GPU_MAX_RANK);

    // Compute strides
    gpu_t->strides[ndim - 1] = 1;
    for (int i = ndim - 2; i >= 0; i--) {
        gpu_t->strides[i] = gpu_t->strides[i + 1] * shape[i + 1];
    }
    gpu_t->size = size;
    gpu_t->capacity = size * sizeof(float);
    gpu_t->owns_data = 0; // Workspace owns memory, not tensor.
    gpu_t->device_id = 0;
    return gpu_t;
}
