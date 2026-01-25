/*
 * gpu_optimizer.h
 *
 * GPU Optimizer declarations - mirrors training/optimizer.h for GPU execution
 */
#ifndef GPU_OPTIMIZER_H
#define GPU_OPTIMIZER_H
#include "gpu/gpu_tensor.h"
#include "training/optimizer_internal.h"

#ifdef __cplusplus
extern "C" {
#endif

// Forward declaration to avoid circular dependency
struct GPUNeuralNet;

typedef struct GPUOptimizer {
    OptimizerType type;
    void *optimizer;
    int num_params;
} GPUOptimizer;

typedef struct {
    float learning_rate;
} GPUSGDOptimizer;

typedef struct {
    float learning_rate;
    float beta;      // Default 0.9;
    GPUTensor **d_v; // Velocity tensors, one per-parameter
} GPUMomentumOptimizer;

typedef struct {
    float learning_rate;
    float beta1;     // Default 0.9
    float beta2;     // Default 0.999
    float epsilon;   // Default 1e-8
    GPUTensor **d_m; // First moment estimates
    GPUTensor **d_s; // Second moment estimates
    int timestep;
} GPUAdamOptimizer;

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

// Apply gradients to parameters (call after gpu_nn_backward)
void gpu_optimizer_step(GPUOptimizer *opt, struct GPUNeuralNet *gpu_nn);

// Apply gradients with pre-scaling (fused scaling + update for better performance)
// grad_scale is typically 1.0/batch_size
void gpu_optimizer_step_scaled(GPUOptimizer *opt, struct GPUNeuralNet *gpu_nn, float grad_scale);

// Learning rate access for scheduler integration
void gpu_optimizer_set_lr(GPUOptimizer *opt, float lr);
float gpu_optimizer_get_lr(GPUOptimizer *opt);

#ifdef __cplusplus
}
#endif

#endif /* ifndef GPU_OPTIMIZER_H */
