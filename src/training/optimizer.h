/*
 * optimizer.h
 *
 * Optimizer declarations
 */
#ifndef OPTIMIZER_H
#define OPTIMIZER_H
#include "nn/nn.h"
#include "tensor/tensor.h"

typedef enum {
    OPTIMIZER_SGD,
    OPTIMIZER_MOMENTUM,
    OPTIMIZER_ADAM,
} OptimizerType;

typedef struct Optimizer {
    OptimizerType type;
    float learning_rate;
    int num_params; // Total number of parameter tensors across all layers

    // Momentum Fields
    float beta; // default 0.9
    Tensor **v; // velocity tensors, one per parameter

    // Adam (extension of momentum)
    float beta1;   // default 0.9
    float beta2;   // default 0.999
    float epsilon; // default 1e-8
    Tensor **m;    // first moment estimates, one per parameter
    Tensor **s;    // second moment estimates, one per parameter
    int timestep;
} Optimizer;

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
