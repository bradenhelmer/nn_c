/*
 * optimizer.h
 *
 * Optimizer declarations
 */
#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "../nn/mlp.h"

typedef enum {
    OPTIMIZER_SGD,
    OPTIMIZER_MOMENTUM,
    OPTIMIZER_ADAM,
} OptimizerType;

typedef struct Optimizer {
    OptimizerType type;
    float learning_rate;
    float num_layers;

    // Momentum Fields
    float beta;         // default 0.9
    Matrix **v_weights; // velocity matrices, one per layer (same shape as weights)
    Vector **v_biases;  // velocity biases, one per layer (same shape as biases)

    // Adam (extension of momentum)
    float beta1;        // default 0.9
    float beta2;        // default 0.999
    float epsilon;      // default 1e-8
    Matrix **m_weights; // first moment
    Matrix **s_weights; // second moment
    Vector **m_biases;
    Vector **s_biases;
    int timestep;
} Optimizer;

// Creation
Optimizer *optimizer_create_sgd(float learning_rate);
Optimizer *optimizer_create_momentum(float learning_rate, float beta);
Optimizer *optimizer_create_adam(float learning_rate, float beta, float beta1, float beta2,
                                 float epsilon, int timestep);
void optimizer_free(Optimizer *opt);

// Call after MLP is built, before training
void optimizer_init(Optimizer *opt, MLP *mlp);

// Apply gradients (call after mlp_backward)
void optimizer_step(Optimizer *opt, MLP *mlp);

// Learning rate access for scheduler integration
void optimizer_set_lr(Optimizer *opt, float lr);
float optimizer_get_lr(Optimizer *opt);

#endif /* ifndef OPTIMIZER_H */
