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

typedef Vector *(*mlp_classifier)(Vector *);

typedef struct {
    Layer **layers;
    int num_layers;
    float learning_rate;
    VectorLossPair loss;
    mlp_classifier classifier;
} MLP;

// Lifecycle
MLP *mlp_create(int num_layers, float learning_rate, VectorLossPair loss_pair,
                mlp_classifier classifier);
void mlp_add_layer(MLP *mlp, int index, Layer *layer);
void mlp_free(MLP *mlp);

// Forward/Backward
Vector *mlp_forward(MLP *mlp, const Vector *input);
void mlp_backward(MLP *mlp, const Vector *target);
float mlp_loss(MLP *mlp, const Vector *target);

// Training
void mlp_zero_gradients(MLP *mlp);
void mlp_update_weights(MLP *mlp);

// Convenience
MLP *mlp_create_sequential(int *layer_sizes, ScalarActivationPair *activations, int num_layers);

// Testing
void test_mlp_on_dataset(MLP *mlp, Dataset *data, const char *name);

#endif /* ifndef MLP_H */
