/*
 * mlp.h
 *
 * Multi-layer (perceptron) network definitions.
 */

#ifndef MLP_H
#define MLP_H
#include "../data/dataset.h"
#include "layer.h"
#include "loss.h"

typedef void (*mlp_classifier)(Tensor *, const Tensor *);

typedef struct {
    LinearLayer **layers;
    int num_layers;
    float learning_rate;
    TensorLossPair loss;
    mlp_classifier classifier;
} MLP;

// Lifecycle
MLP *mlp_create(int num_layers, float learning_rate, TensorLossPair loss_pair,
                mlp_classifier classifier);
void mlp_add_layer(MLP *mlp, int index, LinearLayer *layer);
void mlp_free(MLP *mlp);

// Forward/Backward
Tensor *mlp_forward(MLP *mlp, const Tensor *input);
void mlp_backward(MLP *mlp, const Tensor *target);
float mlp_loss(MLP *mlp, const Tensor *target);

// Training
void mlp_zero_gradients(MLP *mlp);
void mlp_update_weights(MLP *mlp);
void mlp_scale_gradients(MLP *mlp, float scale);
void mlp_add_l2_gradient(MLP *mlp, float lambda);

// Testing
void test_mlp_on_dataset(MLP *mlp, Dataset *data, const char *name);

#endif /* ifndef MLP_H */
