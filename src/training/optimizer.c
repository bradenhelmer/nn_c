/*
 * optimizer.c
 *
 * Optimizer implementations
 */
#include "optimizer.h"
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

Optimizer *optimizer_create_adam(float learning_rate, float beta, float beta1, float beta2,
                                 float epsilon, int timestep) {
    Optimizer *opt = optimizer_create_base(learning_rate);
    opt->type = OPTIMIZER_MOMENTUM;
    opt->beta = beta;
    opt->beta1 = beta1;
    opt->beta2 = beta2;
    opt->epsilon = epsilon;
    opt->timestep = timestep;
    return opt;
}

void optimizer_init(Optimizer *opt, MLP *mlp) {
    opt->num_layers = mlp->num_layers;
    if (opt->type == OPTIMIZER_MOMENTUM || opt->type == OPTIMIZER_ADAM) {
        opt->v_weights = (Matrix **)malloc(sizeof(Matrix *) * mlp->num_layers);
        opt->v_biases = (Vector **)malloc(sizeof(Vector *) * mlp->num_layers);

        for (int i = 0; i < mlp->num_layers; ++i) {
            Layer *layer = mlp->layers[i];

            opt->v_weights[i] = matrix_create(layer->weights->rows, layer->weights->cols);
            opt->v_biases[i] = vector_create(layer->biases->size);
        }

        if (opt->type == OPTIMIZER_ADAM) {
            opt->m_weights = (Matrix **)malloc(sizeof(Matrix *) * mlp->num_layers);
            opt->s_weights = (Matrix **)malloc(sizeof(Matrix *) * mlp->num_layers);
            opt->m_biases = (Vector **)malloc(sizeof(Vector *) * mlp->num_layers);
            opt->s_biases = (Vector **)malloc(sizeof(Vector *) * mlp->num_layers);

            for (int i = 0; i < mlp->num_layers; ++i) {
                Layer *layer = mlp->layers[i];
                opt->m_weights[i] = matrix_create(layer->weights->rows, layer->weights->cols);
                opt->s_weights[i] = matrix_create(layer->weights->rows, layer->weights->cols);
                opt->m_biases[i] = vector_create(layer->biases->size);
                opt->s_biases[i] = vector_create(layer->biases->size);
            }
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
            matrix_free(opt->v_weights[i]);
            vector_free(opt->v_biases[i]);
        }
        free(opt->v_weights);
        free(opt->v_biases);
        break;
    }
    case OPTIMIZER_ADAM: {
        for (int i = 0; i < opt->num_layers; ++i) {
            matrix_free(opt->v_weights[i]);
            vector_free(opt->v_biases[i]);
            matrix_free(opt->m_weights[i]);
            vector_free(opt->m_biases[i]);
            matrix_free(opt->s_weights[i]);
            vector_free(opt->s_biases[i]);
        }
        free(opt->v_weights);
        free(opt->v_biases);
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
        Layer *layer = mlp->layers[i];

        // 1. Update weights -> W = W - (lr * dW)
        matrix_scale(layer->dW, layer->dW, opt->learning_rate);
        matrix_subtract(layer->weights, layer->weights, layer->dW);

        // 2. Update biases -> b = b - (lr * db)
        vector_scale(layer->db, layer->db, opt->learning_rate);
        vector_subtract(layer->biases, layer->biases, layer->db);
    }
}

static void step_momentum(Optimizer *opt, MLP *mlp) {
    for (int i = 0; i < opt->num_layers; ++i) {
        Layer *layer = mlp->layers[i];
        Matrix *vW = opt->v_weights[i];
        Vector *vb = opt->v_biases[i];

        // Update weights: vW = beta * vW + dW, then W = W - lr * vW
        matrix_scale(vW, vW, opt->beta);
        matrix_add(vW, vW, layer->dW);
        Matrix *scaled_vW = matrix_create(vW->rows, vW->cols);
        matrix_scale(scaled_vW, vW, opt->learning_rate);
        matrix_subtract(layer->weights, layer->weights, scaled_vW);
        matrix_free(scaled_vW);

        // Update biases: vb = beta * vb + db, then b = b - lr * vb
        vector_scale(vb, vb, opt->beta);
        vector_add(vb, vb, layer->db);
        Vector *scaled_vb = vector_create(vb->size);
        vector_scale(scaled_vb, vb, opt->learning_rate);
        vector_subtract(layer->biases, layer->biases, scaled_vb);
        vector_free(scaled_vb);
    }
}

static void step_adam(Optimizer *opt, MLP *mlp) {
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
