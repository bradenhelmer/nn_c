/*
 * optimizer.c
 *
 * Optimizer implementations
 */
#include "optimizer.h"
#include <math.h>
#include <stdlib.h>

static Optimizer *optimizer_create_base(float learning_rate) {
    Optimizer *opt = (Optimizer *)malloc(sizeof(Optimizer));
    opt->learning_rate = learning_rate;
    opt->num_layers = 0;
    opt->beta = 0.0f;
    opt->v_weights = NULL;
    opt->v_biases = NULL;
    opt->beta1 = 0.0f;
    opt->beta2 = 0.0f;
    opt->epsilon = 0.0f;
    opt->m_weights = NULL;
    opt->s_weights = NULL;
    opt->m_biases = NULL;
    opt->s_biases = NULL;
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

void optimizer_init(Optimizer *opt, MLP *mlp) {
    opt->num_layers = mlp->num_layers;
    if (opt->type == OPTIMIZER_MOMENTUM) {
        opt->v_weights = (Tensor **)malloc(sizeof(Tensor *) * mlp->num_layers);
        opt->v_biases = (Tensor **)malloc(sizeof(Tensor *) * mlp->num_layers);

        for (int i = 0; i < mlp->num_layers; ++i) {
            LinearLayer *layer = mlp->layers[i];
            opt->v_weights[i] = tensor_clone(layer->weights);
            tensor_fill(opt->v_weights[i], 0.0f);
            opt->v_biases[i] = tensor_clone(layer->biases);
            tensor_fill(opt->v_biases[i], 0.0f);
        }
    }

    if (opt->type == OPTIMIZER_ADAM) {
        opt->m_weights = (Tensor **)malloc(sizeof(Tensor *) * mlp->num_layers);
        opt->s_weights = (Tensor **)malloc(sizeof(Tensor *) * mlp->num_layers);
        opt->m_biases = (Tensor **)malloc(sizeof(Tensor *) * mlp->num_layers);
        opt->s_biases = (Tensor **)malloc(sizeof(Tensor *) * mlp->num_layers);

        for (int i = 0; i < mlp->num_layers; ++i) {
            LinearLayer *layer = mlp->layers[i];
            opt->m_weights[i] = tensor_clone(layer->weights);
            tensor_fill(opt->m_weights[i], 0.0f);
            opt->s_weights[i] = tensor_clone(layer->weights);
            tensor_fill(opt->s_weights[i], 0.0f);
            opt->m_biases[i] = tensor_clone(layer->biases);
            tensor_fill(opt->m_biases[i], 0.0f);
            opt->s_biases[i] = tensor_clone(layer->biases);
            tensor_fill(opt->s_biases[i], 0.0f);
        }
    }
}

void optimizer_free(Optimizer *opt) {
    switch (opt->type) {
    case OPTIMIZER_SGD: {
        break;
    }
    case OPTIMIZER_MOMENTUM: {
        for (int i = 0; i < opt->num_layers; ++i) {
            tensor_free(opt->v_weights[i]);
            tensor_free(opt->v_biases[i]);
        }
        free(opt->v_weights);
        free(opt->v_biases);
        break;
    }
    case OPTIMIZER_ADAM: {
        for (int i = 0; i < opt->num_layers; ++i) {
            tensor_free(opt->m_weights[i]);
            tensor_free(opt->m_biases[i]);
            tensor_free(opt->s_weights[i]);
            tensor_free(opt->s_biases[i]);
        }
        free(opt->m_weights);
        free(opt->m_biases);
        free(opt->s_weights);
        free(opt->s_biases);
        break;
    }
    }

    free(opt);
}

void optimizer_set_lr(Optimizer *opt, float lr) {
    opt->learning_rate = lr;
}

float optimizer_get_lr(Optimizer *opt) {
    return opt->learning_rate;
}

static void step_sgd(Optimizer *opt, MLP *mlp) {
    for (int i = 0; i < mlp->num_layers; i++) {
        LinearLayer *layer = mlp->layers[i];

        // Update weights: W = W - (lr * dW)
        for (int j = 0; j < layer->weights->size; ++j) {
            layer->weights->data[j] -= opt->learning_rate * layer->dW->data[j];
        }

        // Update biases: b = b - (lr * db)
        for (int j = 0; j < layer->biases->size; ++j) {
            layer->biases->data[j] -= opt->learning_rate * layer->db->data[j];
        }
    }
}

static void step_momentum(Optimizer *opt, MLP *mlp) {
    for (int i = 0; i < opt->num_layers; ++i) {
        LinearLayer *layer = mlp->layers[i];
        Tensor *vW = opt->v_weights[i];
        Tensor *vb = opt->v_biases[i];

        // Update weights: vW = beta * vW + dW, then W = W - lr * vW
        for (int j = 0; j < vW->size; ++j) {
            vW->data[j] = opt->beta * vW->data[j] + layer->dW->data[j];
            layer->weights->data[j] -= opt->learning_rate * vW->data[j];
        }

        // Update biases: vb = beta * vb + db, then b = b - lr * vb
        for (int j = 0; j < vb->size; ++j) {
            vb->data[j] = opt->beta * vb->data[j] + layer->db->data[j];
            layer->biases->data[j] -= opt->learning_rate * vb->data[j];
        }
    }
}

static void step_adam(Optimizer *opt, MLP *mlp) {
    opt->timestep += 1;

    // Precompute bias corrections: (1 - β^t)
    float bc1 = 1 - powf(opt->beta1, opt->timestep);
    float bc2 = 1 - powf(opt->beta2, opt->timestep);

    for (int i = 0; i < opt->num_layers; ++i) {
        LinearLayer *layer = mlp->layers[i];
        Tensor *mW = opt->m_weights[i];
        Tensor *sW = opt->s_weights[i];
        Tensor *mb = opt->m_biases[i];
        Tensor *sb = opt->s_biases[i];

        // Update weights: m = β₁m + (1-β₁)g, s = β₂s + (1-β₂)g², W = W - lr·m̂/(√ŝ + ε)
        for (int j = 0; j < layer->weights->size; j++) {
            float grad = layer->dW->data[j];

            // Update moments
            mW->data[j] = opt->beta1 * mW->data[j] + (1 - opt->beta1) * grad;
            sW->data[j] = opt->beta2 * sW->data[j] + (1 - opt->beta2) * grad * grad;

            // Bias correction and parameter update
            float m_hat = mW->data[j] / bc1;
            float s_hat = sW->data[j] / bc2;
            layer->weights->data[j] -= opt->learning_rate * m_hat / (sqrtf(s_hat) + opt->epsilon);
        }

        // Update biases: m = β₁m + (1-β₁)g, s = β₂s + (1-β₂)g², b = b - lr·m̂/(√ŝ + ε)
        for (int j = 0; j < layer->biases->size; j++) {
            float grad = layer->db->data[j];

            // Update moments
            mb->data[j] = opt->beta1 * mb->data[j] + (1 - opt->beta1) * grad;
            sb->data[j] = opt->beta2 * sb->data[j] + (1 - opt->beta2) * grad * grad;

            // Bias correction and parameter update
            float m_hat = mb->data[j] / bc1;
            float s_hat = sb->data[j] / bc2;
            layer->biases->data[j] -= opt->learning_rate * m_hat / (sqrtf(s_hat) + opt->epsilon);
        }
    }
}

void optimizer_step(Optimizer *opt, MLP *mlp) {
    switch (opt->type) {
    case OPTIMIZER_SGD:
        step_sgd(opt, mlp);
        break;
    case OPTIMIZER_MOMENTUM:
        step_momentum(opt, mlp);
        break;
    case OPTIMIZER_ADAM:
        step_adam(opt, mlp);
        break;
    }
}
