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

typedef struct Optimizer Optimizer;

// Creation
Optimizer *optimizer_create_sgd(float learning_rate);
Optimizer *optimizer_create_momentum(float learning_rate);
Optimizer *optimizer_create_adam(float learning_rate);
void optimizer_free(Optimizer *opt);

// Call after MLP is built, before training
void optimizer_init(Optimizer *opt, MLP *mlp);

// Apply gradients (call after mlp_backward)
void optimizer_step(Optimizer *opt, MLP *mlp);

// Learning rate access for scheduler integration
void optimizer_set_lr(Optimizer *opt, float lr);
float optimizer_get_lr(Optimizer *opt);

#endif /* ifndef OPTIMIZER_H */
