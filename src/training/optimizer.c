/*
 * optimizer.c
 *
 * Optimizer implementations
 */
#include "optimizer.h"
#include "nn/layer.h"
#include <math.h>
#include <stdlib.h>

static Optimizer *optimizer_create_base(float learning_rate) {
    Optimizer *opt = (Optimizer *)malloc(sizeof(Optimizer));
    opt->learning_rate = learning_rate;
    opt->num_params = 0;
    opt->beta = 0.0f;
    opt->v = NULL;
    opt->beta1 = 0.0f;
    opt->beta2 = 0.0f;
    opt->epsilon = 0.0f;
    opt->m = NULL;
    opt->s = NULL;
    opt->timestep = 0;

    return opt;
}

Optimizer *optimizer_create_sgd(float learning_rate) {
    Optimizer *opt = optimizer_create_base(learning_rate);
    opt->type = OPTIMIZER_SGD;
    return opt;
}

Optimizer *optimizer_create_momentum(float learning_rate, float beta) {
    Optimizer *opt = optimizer_create_base(learning_rate);
    opt->type = OPTIMIZER_MOMENTUM;
    opt->beta = beta;
    return opt;
}

Optimizer *optimizer_create_adam(float learning_rate, float beta1, float beta2, float epsilon) {
    Optimizer *opt = optimizer_create_base(learning_rate);
    opt->type = OPTIMIZER_ADAM;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->epsilon = epsilon;
    opt->timestep = 0;
    return opt;
}

void optimizer_init(Optimizer *opt, NeuralNet *nn) {
    // Count total parameters across all layers
    int total_params = 0;
    for (int i = 0; i < nn->num_layers; i++) {
        LayerParameters params = layer_get_parameters(nn->layers[i]);
        total_params += params.num_pairs;
        layer_parameters_free(&params);
    }

    opt->num_params = total_params;

    // Allocate momentum arrays if needed
    if (opt->type == OPTIMIZER_MOMENTUM) {
        opt->v = (Tensor **)malloc(sizeof(Tensor *) * total_params);

        int param_idx = 0;
        for (int i = 0; i < nn->num_layers; i++) {
            LayerParameters params = layer_get_parameters(nn->layers[i]);
            for (int j = 0; j < params.num_pairs; j++) {
                opt->v[param_idx] = tensor_clone(params.pairs[j].param);
                tensor_fill(opt->v[param_idx], 0.0f);
                param_idx++;
            }
            layer_parameters_free(&params);
        }
    }

    // Allocate Adam arrays if needed
    if (opt->type == OPTIMIZER_ADAM) {
        opt->m = (Tensor **)malloc(sizeof(Tensor *) * total_params);
        opt->s = (Tensor **)malloc(sizeof(Tensor *) * total_params);

        int param_idx = 0;
        for (int i = 0; i < nn->num_layers; i++) {
            LayerParameters params = layer_get_parameters(nn->layers[i]);
            for (int j = 0; j < params.num_pairs; j++) {
                opt->m[param_idx] = tensor_clone(params.pairs[j].param);
                tensor_fill(opt->m[param_idx], 0.0f);
                opt->s[param_idx] = tensor_clone(params.pairs[j].param);
                tensor_fill(opt->s[param_idx], 0.0f);
                param_idx++;
            }
            layer_parameters_free(&params);
        }
    }
}

void optimizer_free(Optimizer *opt) {
    switch (opt->type) {
    case OPTIMIZER_SGD:
        // No state to free
        break;
    case OPTIMIZER_MOMENTUM:
        for (int i = 0; i < opt->num_params; ++i) {
            tensor_free(opt->v[i]);
        }
        free(opt->v);
        break;
    case OPTIMIZER_ADAM:
        for (int i = 0; i < opt->num_params; ++i) {
            tensor_free(opt->m[i]);
            tensor_free(opt->s[i]);
        }
        free(opt->m);
        free(opt->s);
        break;
    }

    free(opt);
}

void optimizer_set_lr(Optimizer *opt, float lr) {
    opt->learning_rate = lr;
}

float optimizer_get_lr(Optimizer *opt) {
    return opt->learning_rate;
}

static void step_sgd(Optimizer *opt, NeuralNet *nn) {
    for (int i = 0; i < nn->num_layers; i++) {
        layer_update_weights(nn->layers[i], opt->learning_rate);
    }
}

static void step_momentum(Optimizer *opt, NeuralNet *nn) {
    int param_idx = 0;

    for (int i = 0; i < nn->num_layers; i++) {
        LayerParameters params = layer_get_parameters(nn->layers[i]);

        for (int j = 0; j < params.num_pairs; j++) {
            Tensor *param = params.pairs[j].param;
            Tensor *grad = params.pairs[j].grad_param;
            Tensor *v = opt->v[param_idx];

            // Update: v = beta * v + grad, then param = param - lr * v
            for (int k = 0; k < param->size; k++) {
                v->data[k] = opt->beta * v->data[k] + grad->data[k];
                param->data[k] -= opt->learning_rate * v->data[k];
            }

            param_idx++;
        }

        layer_parameters_free(&params);
    }
}

static void step_adam(Optimizer *opt, NeuralNet *nn) {
    opt->timestep += 1;

    // Precompute bias corrections: (1 - β^t)
    float bc1 = 1 - powf(opt->beta1, opt->timestep);
    float bc2 = 1 - powf(opt->beta2, opt->timestep);

    int param_idx = 0;

    for (int i = 0; i < nn->num_layers; i++) {
        LayerParameters params = layer_get_parameters(nn->layers[i]);

        for (int j = 0; j < params.num_pairs; j++) {
            Tensor *param = params.pairs[j].param;
            Tensor *grad = params.pairs[j].grad_param;
            Tensor *m = opt->m[param_idx];
            Tensor *s = opt->s[param_idx];

            // Update: m = β₁m + (1-β₁)g, s = β₂s + (1-β₂)g², param = param - lr·m̂/(√ŝ + ε)
            for (int k = 0; k < param->size; k++) {
                float g = grad->data[k];

                // Update biased first moment estimate
                m->data[k] = opt->beta1 * m->data[k] + (1 - opt->beta1) * g;

                // Update biased second raw moment estimate
                s->data[k] = opt->beta2 * s->data[k] + (1 - opt->beta2) * g * g;

                // Compute bias-corrected estimates
                float m_hat = m->data[k] / bc1;
                float s_hat = s->data[k] / bc2;

                // Update parameters
                param->data[k] -= opt->learning_rate * m_hat / (sqrtf(s_hat) + opt->epsilon);
            }

            param_idx++;
        }

        layer_parameters_free(&params);
    }
}

void optimizer_step(Optimizer *opt, NeuralNet *nn) {
    switch (opt->type) {
    case OPTIMIZER_SGD:
        step_sgd(opt, nn);
        break;
    case OPTIMIZER_MOMENTUM:
        step_momentum(opt, nn);
        break;
    case OPTIMIZER_ADAM:
        step_adam(opt, nn);
        break;
    }
}
