/*
 * optimizer_internal.h - Internal optimizer struct definitions.
 */

#ifndef OPTIMIZER_INTERNAL_H
#define OPTIMIZER_INTERNAL_H
#include "optimizer.h"
#include "tensor/tensor.h"
#include <stdlib.h>

// ==============================================================================
// OPTIMIZER GENERICS
// ==============================================================================

// Optimizer types.
typedef enum {
    OPTIMIZER_SGD,
    OPTIMIZER_MOMENTUM,
    OPTIMIZER_ADAM,
} OptimizerType;

// Core optimizer struct.
struct Optimizer {
    OptimizerType type;
    void *optimizer;

    int num_params;
};

static Optimizer *optimizer_create(OptimizerType type, void *optimizer) {
    Optimizer *O = (Optimizer *)malloc(sizeof(Optimizer));
    O->type = type;
    O->optimizer = optimizer;
    O->num_params = 0;
    return O;
}

// ==============================================================================
// SGD OPTIMIZER
// ==============================================================================

struct SGDOptimizer {
    float learning_rate;
};

// ==============================================================================
// MOMENTUM OPTIMIZER
// ==============================================================================

struct MomentumOptimizer {
    float learning_rate;
    float beta; // Default 0.9;
    Tensor **v; // Velocity tensors, one per-parameter
};

// ==============================================================================
// ADAM OPTIMIZER
// ==============================================================================

struct AdamOptimizer {
    float learning_rate;
    float beta1;   // Default 0.9
    float beta2;   // Default 0.999
    float epsilon; // Default 1e-8
    Tensor **m;    // First moment estimates
    Tensor **s;    // Second moment estimates
    int timestep;
};

#endif /* ifndef OPTIMIZER_INTERNAL_H */
