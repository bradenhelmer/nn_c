/*
 * optimizer.c
 *
 * Optimizer implementations
 */
#include "nn/layer_internal.h"
#include "nn/nn.h"
#include "optimizer_internal.h"
#include "tensor/tensor_internal.h"
#include <math.h>
#include <stdlib.h>

Optimizer *optimizer_create_sgd(float learning_rate) {
    SGDOptimizer *sgd_opt = (SGDOptimizer *)malloc(sizeof(SGDOptimizer));
    sgd_opt->learning_rate = learning_rate;
    return optimizer_create(OPTIMIZER_SGD, (void *)sgd_opt);
}

Optimizer *optimizer_create_momentum(float learning_rate, float beta) {
    MomentumOptimizer *m_opt = (MomentumOptimizer *)malloc(sizeof(MomentumOptimizer));
    m_opt->learning_rate = learning_rate;
    m_opt->beta = beta;
    m_opt->v = NULL;
    return optimizer_create(OPTIMIZER_MOMENTUM, (void *)m_opt);
}

Optimizer *optimizer_create_adam(float learning_rate, float beta1, float beta2, float epsilon) {
    AdamOptimizer *adam_opt = (AdamOptimizer *)malloc(sizeof(AdamOptimizer));
    adam_opt->learning_rate = learning_rate;
    adam_opt->beta1 = beta1;
    adam_opt->beta2 = beta2;
    adam_opt->epsilon = epsilon;
    adam_opt->timestep = 0;
    adam_opt->m = NULL;
    adam_opt->s = NULL;
    return optimizer_create(OPTIMIZER_ADAM, (void *)adam_opt);
}

void optimizer_init(Optimizer *opt, NeuralNet *nn) {
    // Count total parameters across all layers
    int total_params = 0;
    for (int i = 0; i < nn_get_num_layers(nn); i++) {
        LayerParameters params = layer_get_parameters(nn_get_layer(nn, i));
        total_params += params.num_pairs;
        layer_parameters_free(&params);
    }

    opt->num_params = total_params;

    switch (opt->type) {
    case OPTIMIZER_SGD:
        break;
    case OPTIMIZER_MOMENTUM: {
        MomentumOptimizer *m_opt = (MomentumOptimizer *)opt->optimizer;
        m_opt->v = (Tensor **)malloc(sizeof(Tensor *) * total_params);

        int param_idx = 0;
        for (int i = 0; i < nn_get_num_layers(nn); i++) {
            LayerParameters params = layer_get_parameters(nn_get_layer(nn, i));
            for (int j = 0; j < params.num_pairs; j++) {
                m_opt->v[param_idx] = tensor_clone(params.pairs[j].param);
                tensor_fill(m_opt->v[param_idx], 0.0f);
                param_idx++;
            }
            layer_parameters_free(&params);
        }
        break;
    }
    case OPTIMIZER_ADAM: {
        AdamOptimizer *adam_opt = (AdamOptimizer *)opt->optimizer;
        adam_opt->m = (Tensor **)malloc(sizeof(Tensor *) * total_params);
        adam_opt->s = (Tensor **)malloc(sizeof(Tensor *) * total_params);

        int param_idx = 0;
        for (int i = 0; i < nn_get_num_layers(nn); i++) {
            LayerParameters params = layer_get_parameters(nn_get_layer(nn, i));
            for (int j = 0; j < params.num_pairs; j++) {
                adam_opt->m[param_idx] = tensor_clone(params.pairs[j].param);
                tensor_fill(adam_opt->m[param_idx], 0.0f);
                adam_opt->s[param_idx] = tensor_clone(params.pairs[j].param);
                tensor_fill(adam_opt->s[param_idx], 0.0f);
                param_idx++;
            }
            layer_parameters_free(&params);
        }
        break;
    }
    }
}

void optimizer_free(Optimizer *opt) {
    switch (opt->type) {
    case OPTIMIZER_SGD: {
        free(opt->optimizer);
        break;
    }
    case OPTIMIZER_MOMENTUM: {
        MomentumOptimizer *m_opt = (MomentumOptimizer *)opt->optimizer;
        for (int i = 0; i < opt->num_params; ++i) {
            tensor_free(m_opt->v[i]);
        }
        free(m_opt->v);
        free(m_opt);
        break;
    }
    case OPTIMIZER_ADAM: {
        AdamOptimizer *adam_opt = (AdamOptimizer *)opt->optimizer;
        for (int i = 0; i < opt->num_params; ++i) {
            tensor_free(adam_opt->m[i]);
            tensor_free(adam_opt->s[i]);
        }
        free(adam_opt->m);
        free(adam_opt->s);
        free(adam_opt);
        break;
    }
    }

    free(opt);
}

void optimizer_set_lr(Optimizer *opt, float lr) {
    switch (opt->type) {
    case OPTIMIZER_SGD: {
        ((SGDOptimizer *)opt->optimizer)->learning_rate = lr;
        break;
    }
    case OPTIMIZER_MOMENTUM: {
        ((MomentumOptimizer *)opt->optimizer)->learning_rate = lr;
        break;
    }
    case OPTIMIZER_ADAM: {
        ((AdamOptimizer *)opt->optimizer)->learning_rate = lr;
        break;
    }
    }
}

float optimizer_get_lr(Optimizer *opt) {
    switch (opt->type) {
    case OPTIMIZER_SGD: {
        return ((SGDOptimizer *)opt->optimizer)->learning_rate;
    }
    case OPTIMIZER_MOMENTUM: {
        return ((MomentumOptimizer *)opt->optimizer)->learning_rate;
    }
    case OPTIMIZER_ADAM: {
        return ((AdamOptimizer *)opt->optimizer)->learning_rate;
    }
    }
}

static void step_sgd(SGDOptimizer *sgd_opt, NeuralNet *nn) {
    for (int i = 0; i < nn_get_num_layers(nn); i++) {
        layer_update_weights(nn_get_layer(nn, i), sgd_opt->learning_rate);
    }
}

static void step_momentum(MomentumOptimizer *opt, NeuralNet *nn) {
    int param_idx = 0;

    for (int i = 0; i < nn_get_num_layers(nn); i++) {
        LayerParameters params = layer_get_parameters(nn_get_layer(nn, i));

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

static void step_adam(AdamOptimizer *opt, NeuralNet *nn) {
    opt->timestep += 1;

    // Precompute bias corrections: (1 - β^t)
    float bc1 = 1 - powf(opt->beta1, opt->timestep);
    float bc2 = 1 - powf(opt->beta2, opt->timestep);

    int param_idx = 0;

    for (int i = 0; i < nn_get_num_layers(nn); i++) {
        LayerParameters params = layer_get_parameters(nn_get_layer(nn, i));

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
        step_sgd((SGDOptimizer *)opt->optimizer, nn);
        break;
    case OPTIMIZER_MOMENTUM:
        step_momentum((MomentumOptimizer *)opt->optimizer, nn);
        break;
    case OPTIMIZER_ADAM:
        step_adam((AdamOptimizer *)opt->optimizer, nn);
        break;
    }
}
