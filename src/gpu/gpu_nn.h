/*
 * gpu_nn.h - GPUNeuralNet executor declarations.
 */
#ifndef GPU_NN_H
#define GPU_NN_H
#include "data/dataset.h"
#include "gpu/gpu_tensor.h"
#include "nn/nn.h"
#include <cublas_v2.h>

// Input shape descriptor for dynamic workspace sizing
typedef struct {
    int height;   // Image height (1 for fully-connected networks)
    int width;    // Image width (1 for fully-connected networks)
    int channels; // Number of channels (or feature size for FC)
} InputShape;

typedef struct {
    NeuralNet *cpu_nn; // Borrowed for configuration (read only)
    int num_layers;
    int batch_size; // Fixed at creation
    float learning_rate;
    InputShape input_shape;

    // Parameter storage (indexed by layer)
    GPUTensor **d_params;    // Device parameters
    GPUTensor **d_grads;     // Device gradients
    int num_params;          // Total parameter tensors
    int *layer_param_offset; // layer_param_idx[2] = 4 means layer 2's weights start at d_params[4]

    // Input/Output Cache (one per layer)
    GPUTensor **d_inputs;  // Input to each layer for backward pass
    GPUTensor **d_outputs; // Output of each layer for backward pass

    // LAYER_CONV_2D:   NULL (im2col from workspace)
    // LAYER_MAX_POOL:  int* d_max_indices
    // LAYER_FLATTEN:   int* original_shape (host memory)
    // Others:          NULL
    void **layer_aux; // Pre-layer auxiliary storage.

    // Workspace
    float *d_workspace;      // Single large allocation
    size_t workspace_size;   // Total bytes
    size_t workspace_offset; // Current bump pointer

    cublasHandle_t cublas;
    cudaStream_t compute_stream;
    cudaStream_t transfer_stream; // For async Host <-> Device

    // Optimizer state (ADAM)
    GPUTensor **d_m; // First moment per param.
    GPUTensor **d_v; // Second moment per param.
    int t;           // Timestep counter
    float beta1, beta2, eps;
} GPUNeuralNet;

// Lifecyle
GPUNeuralNet *gpu_nn_create_from_cpu_nn(NeuralNet *cpu_nn, int batch_size, InputShape input_shape);
void gpu_nn_to_cpu_nn(GPUNeuralNet *gpu_nn);
void gpu_nn_free(GPUNeuralNet *gpu_nn);

// Training
GPUTensor *gpu_nn_forward(GPUNeuralNet *gpu_nn, GPUTensor *input);
void gpu_nn_backward(GPUNeuralNet *gpu_nn, GPUTensor *target);
void gpu_nn_zero_gradients(GPUNeuralNet *gpu_nn);
void gpu_nn_scale_gradients(GPUNeuralNet *gpu_nn, float scale);
float gpu_nn_compute_loss(GPUNeuralNet *gpu_nn, GPUTensor *prediction, GPUTensor *target);
void gpu_nn_optimizer_step(GPUNeuralNet *gpu_nn);

// Evaluation
void gpu_nn_predict(GPUNeuralNet *gpu_nn, Tensor *host_input, Tensor *host_output, int batch_size);
float gpu_nn_evaluate_accuracy(GPUNeuralNet *gpu_nn, Dataset *val_data, GPUTensor *d_input,
                               float *h_input_pinned);

// Workspace functions
void workspace_reset(GPUNeuralNet *gpu_nn);
float *workspace_alloc(GPUNeuralNet *gpu_nn, size_t bytes);
GPUTensor *workspace_alloc_tensor(GPUNeuralNet *gpu_nn, int ndim, int shape[GPU_MAX_RANK]);

#endif /* ifndef GPU_NN_H */
