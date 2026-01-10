/*
 * gpu_optimizer.h
 *
 * GPU Optimizer declarations - mirrors training/optimizer.h for GPU execution
 */
#ifndef GPU_OPTIMIZER_H
#define GPU_OPTIMIZER_H
#include "gpu/gpu_tensor.h"
#include "training/optimizer.h"

// Forward declaration to avoid circular dependency
struct GPUNeuralNet;

typedef struct GPUOptimizer {
    OptimizerType type;
    float learning_rate;
    int num_params; // Total number of parameter tensors across all layers

    // Momentum fields
    float beta;      // default 0.9
    GPUTensor **d_v; // velocity tensors on device, one per parameter

    // Adam fields (extension of momentum)
    float beta1;     // default 0.9
    float beta2;     // default 0.999
    float epsilon;   // default 1e-8
    GPUTensor **d_m; // first moment estimates on device, one per parameter
    GPUTensor **d_s; // second moment estimates on device, one per parameter
    int timestep;
} GPUOptimizer;

// Creation - mirrors CPU optimizer creation functions
GPUOptimizer *gpu_optimizer_create_sgd(float learning_rate);
GPUOptimizer *gpu_optimizer_create_momentum(float learning_rate, float beta);
GPUOptimizer *gpu_optimizer_create_adam(float learning_rate, float beta1, float beta2,
                                        float epsilon);

// Initialize with network structure - allocates GPU tensors for state
// Must be called after gpu_nn is created
void gpu_optimizer_init(GPUOptimizer *opt, struct GPUNeuralNet *gpu_nn);

// Cleanup
void gpu_optimizer_free(GPUOptimizer *opt);

// Learning rate access for scheduler integration
void gpu_optimizer_set_lr(GPUOptimizer *opt, float lr);
float gpu_optimizer_get_lr(GPUOptimizer *opt);

#endif /* ifndef GPU_OPTIMIZER_H */
