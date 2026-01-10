/*
 * gpu_optimizer.c
 *
 * GPU Optimizer implementations
 */
#include "gpu_optimizer.h"
#include "gpu_nn.h"
#include "gpu_tensor.h"
#include <stdlib.h>

static GPUOptimizer *gpu_optimizer_create_base(float learning_rate) {
    GPUOptimizer *opt = (GPUOptimizer *)malloc(sizeof(GPUOptimizer));
    opt->learning_rate = learning_rate;
    opt->num_params = 0;
    opt->d_v = NULL;
    opt->d_m = NULL;
    opt->d_s = NULL;
    opt->timestep = 0;
    return opt;
}

GPUOptimizer *gpu_optimizer_create_sgd(float learning_rate) {
    GPUOptimizer *opt = gpu_optimizer_create_base(learning_rate);
    opt->type = OPTIMIZER_SGD;
    return opt;
}

GPUOptimizer *gpu_optimizer_create_momentum(float learning_rate, float beta) {
    GPUOptimizer *opt = gpu_optimizer_create_base(learning_rate);
    opt->type = OPTIMIZER_MOMENTUM;
    opt->beta = beta;
    return opt;
}

GPUOptimizer *gpu_optimizer_create_adam(float learning_rate, float beta1, float beta2,
                                        float epsilon) {
    GPUOptimizer *opt = gpu_optimizer_create_base(learning_rate);
    opt->type = OPTIMIZER_ADAM;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->epsilon = epsilon;
    opt->timestep = 0;
    return opt;
}

void gpu_optimizer_init(GPUOptimizer *opt, struct GPUNeuralNet *gpu_nn) {
    int num_params = gpu_nn->num_params;
    opt->num_params = num_params;

    switch (opt->type) {
    case OPTIMIZER_SGD:
        // SGD has no state tensors
        break;

    case OPTIMIZER_MOMENTUM:
        // Allocate velocity tensors
        opt->d_v = (GPUTensor **)malloc(sizeof(GPUTensor *) * num_params);
        for (int i = 0; i < num_params; i++) {
            opt->d_v[i] = gpu_tensor_create_like(gpu_nn->d_params[i]);
        }
        break;

    case OPTIMIZER_ADAM:
        // Allocate first and second moment tensors
        opt->d_m = (GPUTensor **)malloc(sizeof(GPUTensor *) * num_params);
        opt->d_s = (GPUTensor **)malloc(sizeof(GPUTensor *) * num_params);
        for (int i = 0; i < num_params; i++) {
            opt->d_m[i] = gpu_tensor_create_like(gpu_nn->d_params[i]);
            opt->d_s[i] = gpu_tensor_create_like(gpu_nn->d_params[i]);
        }
        break;
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

    case OPTIMIZER_MOMENTUM:
        if (opt->d_v != NULL) {
            for (int i = 0; i < opt->num_params; i++) {
                gpu_tensor_free(opt->d_v[i]);
            }
            free(opt->d_v);
        }
        break;

    case OPTIMIZER_ADAM:
        if (opt->d_m != NULL) {
            for (int i = 0; i < opt->num_params; i++) {
                gpu_tensor_free(opt->d_m[i]);
            }
            free(opt->d_m);
        }
        if (opt->d_s != NULL) {
            for (int i = 0; i < opt->num_params; i++) {
                gpu_tensor_free(opt->d_s[i]);
            }
            free(opt->d_s);
        }
        break;
    }

    free(opt);
}

void gpu_optimizer_set_lr(GPUOptimizer *opt, float lr) {
    opt->learning_rate = lr;
}

float gpu_optimizer_get_lr(GPUOptimizer *opt) {
    return opt->learning_rate;
}
