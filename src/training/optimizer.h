/*
 * optimizer.h
 *
 * Optimizer declarations
 */
#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "nn/nn.h"
#include "core/tensor.h"

typedef struct Optimizer Optimizer;
typedef struct SGDOptimizer SGDOptimizer;
typedef struct MomentumOptimizer MomentumOptimizer;
typedef struct AdamOptimizer AdamOptimizer;

// Creation
Optimizer *optimizer_create_sgd(float learning_rate);
Optimizer *optimizer_create_momentum(float learning_rate, float beta);
Optimizer *optimizer_create_adam(float learning_rate, float beta1, float beta2, float epsilon);

void optimizer_free(Optimizer *opt);

// Call after NN is built, before training
void optimizer_init(Optimizer *opt, NeuralNet *nn);

// Apply gradients (call after mlp_backward)
void optimizer_step(Optimizer *opt, NeuralNet *nn);

// Learning rate access for scheduler integration
void optimizer_set_lr(Optimizer *opt, float lr);
float optimizer_get_lr(Optimizer *opt);

#endif /* ifndef OPTIMIZER_H */
