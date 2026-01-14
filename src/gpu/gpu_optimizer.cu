/*
 * gpu_optimizer.c
 *
 * GPU Optimizer implementations
 */
#include "gpu_nn.h"
#include "gpu_optimizer.h"
#include "gpu_tensor.h"
#include "layers/layer_internal.h"
#include "training/optimizer_internal.h"
#include <stdlib.h>

static GPUOptimizer *_gpu_optimizer_create(OptimizerType type, void *optimizer) {
    GPUOptimizer *opt = (GPUOptimizer *)malloc(sizeof(GPUOptimizer));
    return opt;
}

GPUOptimizer *gpu_optimizer_create_sgd(float learning_rate) {
    GPUSGDOptimizer *sgd_opt = (GPUSGDOptimizer *)malloc(sizeof(GPUSGDOptimizer));
    sgd_opt->learning_rate = learning_rate;
    return _gpu_optimizer_create(OPTIMIZER_SGD, (void *)sgd_opt);
}

GPUOptimizer *gpu_optimizer_create_momentum(float learning_rate, float beta) {
    GPUMomentumOptimizer *m_opt = (GPUMomentumOptimizer *)malloc(sizeof(GPUMomentumOptimizer));
    m_opt->learning_rate = learning_rate;
    m_opt->beta = beta;
    m_opt->d_v = NULL;
    return _gpu_optimizer_create(OPTIMIZER_MOMENTUM, (void *)m_opt);
}

GPUOptimizer *gpu_optimizer_create_adam(float learning_rate, float beta1, float beta2,
                                        float epsilon) {
    GPUAdamOptimizer *adam_opt = (GPUAdamOptimizer *)malloc(sizeof(GPUAdamOptimizer));
    adam_opt->learning_rate = learning_rate;
    adam_opt->beta1 = beta1;
    adam_opt->beta2 = beta2;
    adam_opt->epsilon = epsilon;
    adam_opt->timestep = 0;
    adam_opt->d_m = NULL;
    adam_opt->d_s = NULL;
    return _gpu_optimizer_create(OPTIMIZER_ADAM, (void *)adam_opt);
}

void gpu_optimizer_init(GPUOptimizer *opt, struct GPUNeuralNet *gpu_nn) {
    int num_params = gpu_nn->num_params;
    opt->num_params = num_params;

    switch (opt->type) {
    case OPTIMIZER_SGD:
        // SGD has no state tensors
        break;

    case OPTIMIZER_MOMENTUM: {
        GPUMomentumOptimizer *m_opt = (GPUMomentumOptimizer *)opt->optimizer;
        // Allocate velocity tensors
        m_opt->d_v = (GPUTensor **)malloc(sizeof(GPUTensor *) * num_params);
        for (int i = 0; i < num_params; i++) {
            m_opt->d_v[i] = gpu_tensor_create_like(gpu_nn->d_params[i]);
        }
        break;
    }

    case OPTIMIZER_ADAM: {
        GPUAdamOptimizer *adam_opt = (GPUAdamOptimizer *)opt->optimizer;
        // Allocate first and second moment tensors
        adam_opt->d_m = (GPUTensor **)malloc(sizeof(GPUTensor *) * num_params);
        adam_opt->d_s = (GPUTensor **)malloc(sizeof(GPUTensor *) * num_params);
        for (int i = 0; i < num_params; i++) {
            adam_opt->d_m[i] = gpu_tensor_create_like(gpu_nn->d_params[i]);
            adam_opt->d_s[i] = gpu_tensor_create_like(gpu_nn->d_params[i]);
        }
        break;
    }
    }
}

void gpu_optimizer_free(GPUOptimizer *opt) {
    if (opt == NULL) {
        return;
    }

    switch (opt->type) {
    case OPTIMIZER_SGD:
        // No state to free
        break;
    case OPTIMIZER_MOMENTUM: {
        GPUMomentumOptimizer *m_opt = (GPUMomentumOptimizer *)opt->optimizer;
        if (m_opt->d_v != NULL) {
            for (int i = 0; i < opt->num_params; i++) {
                gpu_tensor_free(m_opt->d_v[i]);
            }
            free(m_opt->d_v);
        }
        break;
    }
    case OPTIMIZER_ADAM: {
        GPUAdamOptimizer *adam_opt = (GPUAdamOptimizer *)opt->optimizer;

        if (adam_opt->d_m != NULL) {
            for (int i = 0; i < opt->num_params; i++) {
                gpu_tensor_free(adam_opt->d_m[i]);
            }
            free(adam_opt->d_m);
        }
        if (adam_opt->d_s != NULL) {
            for (int i = 0; i < opt->num_params; i++) {
                gpu_tensor_free(adam_opt->d_s[i]);
            }
            free(adam_opt->d_s);
        }
        break;
    }
    }

    free(opt);
}

void gpu_optimizer_set_lr(GPUOptimizer *opt, float lr) {
    switch (opt->type) {
    case OPTIMIZER_SGD: {
        ((GPUSGDOptimizer *)opt->optimizer)->learning_rate = lr;
        break;
    }
    case OPTIMIZER_MOMENTUM: {
        ((GPUMomentumOptimizer *)opt->optimizer)->learning_rate = lr;
        break;
    }
    case OPTIMIZER_ADAM: {
        ((GPUAdamOptimizer *)opt->optimizer)->learning_rate = lr;
        break;
    }
    }
}

float gpu_optimizer_get_lr(GPUOptimizer *opt) {
    switch (opt->type) {
    case OPTIMIZER_SGD: {
        return ((GPUSGDOptimizer *)opt->optimizer)->learning_rate;
    }
    case OPTIMIZER_MOMENTUM: {
        return ((GPUMomentumOptimizer *)opt->optimizer)->learning_rate;
    }
    case OPTIMIZER_ADAM: {
        return ((GPUAdamOptimizer *)opt->optimizer)->learning_rate;
    }
    }
}

#define THREADS 256
#define BLOCKS(size) ((size) + (THREADS) - 1) / (THREADS)

__global__ void _sgd_update_kernel(float *weights, float *grad_weights, float learning_rate,
                                           int size) {
    const int tid = threadIdx.x;
    const int global_idx = blockIdx.x * blockDim.x + tid;
    if (global_idx < size) {
        weights[global_idx] -= grad_weights[global_idx] * learning_rate;
    }
}

// Internal step implementations (static - not exposed in header)
static void _gpu_step_sgd(GPUSGDOptimizer *opt, GPUNeuralNet *gpu_nn) {
    // TODO: Implement SGD parameter update
    // For each parameter: param -= learning_rate * grad
    for (int i = 0; i < gpu_nn->num_layers; i++) {
        Layer *layer = gpu_nn->cpu_nn->layers[i];
        switch (layer->type) {
        case LAYER_CONV_2D:
        case LAYER_LINEAR: {
            const int p_idx = gpu_nn->layer_param_offset[i];

            GPUTensor *weights = gpu_nn->d_params[p_idx];
            GPUTensor *grad_weights = gpu_nn->d_grads[p_idx];
            _sgd_update_kernel<<<BLOCKS(weights->size), THREADS>>>(
                weights->d_data, grad_weights->d_data, opt->learning_rate, weights->size);

            GPUTensor *biases = gpu_nn->d_params[p_idx + 1];
            GPUTensor *grad_biases = gpu_nn->d_grads[p_idx + 1];
            _sgd_update_kernel<<<BLOCKS(biases->size), THREADS>>>(
                biases->d_data, grad_biases->d_data, opt->learning_rate, biases->size);

            break;
        }
        case LAYER_ACTIVATION:
        case LAYER_MAX_POOL:
        case LAYER_FLATTEN:
        case LAYER_RESHAPE:
            // No weights to update
            break;
        }
    }
}

static void _gpu_step_momentum(GPUOptimizer *opt, GPUNeuralNet *gpu_nn) {
    // TODO: Implement Momentum parameter update
    // v = beta * v + grad
    // param -= learning_rate * v
    (void)opt;
    (void)gpu_nn;
}

static void _gpu_step_adam(GPUOptimizer *opt, GPUNeuralNet *gpu_nn) {
    // TODO: Implement Adam parameter update
    // m = beta1 * m + (1 - beta1) * grad
    // s = beta2 * s + (1 - beta2) * grad^2
    // m_hat = m / (1 - beta1^t)
    // s_hat = s / (1 - beta2^t)
    // param -= learning_rate * m_hat / (sqrt(s_hat) + epsilon)
    (void)opt;
    (void)gpu_nn;
}

void gpu_optimizer_step(GPUOptimizer *opt, GPUNeuralNet *gpu_nn) {
    switch (opt->type) {
    case OPTIMIZER_SGD:
        _gpu_step_sgd((GPUSGDOptimizer *)opt->optimizer, gpu_nn);
        break;
    case OPTIMIZER_MOMENTUM:
        _gpu_step_momentum(opt, gpu_nn);
        break;
    case OPTIMIZER_ADAM:
        _gpu_step_adam(opt, gpu_nn);
        break;
    }
}
