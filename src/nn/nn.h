/*
 * nn.c
 *
 * Core neural network declarations.
 */

#ifndef NN_H
#define NN_H
#include "../data/dataset.h"
#include "layer.h"
#include "loss.h"

typedef void (*Classifier)(Tensor *, const Tensor *);

typedef struct {
    Layer **layers;
    int num_layers;
    float learning_rate;
    LossType loss_type;
    Classifier classifier;
} NeuralNet;

// Lifecycle
NeuralNet *nn_create(int num_layers, float learning_rate, LossType loss_type,
                     Classifier classifier);
void nn_add_layer(NeuralNet *nn, int index, Layer *layer);
void nn_free(NeuralNet *nn);

// Forward/Backward
Tensor *nn_forward(NeuralNet *nn, const Tensor *input);
void nn_backward(NeuralNet *nn, const Tensor *target);
float nn_loss(NeuralNet *nn, const Tensor *prediction, const Tensor *target);
void nn_loss_derivative(NeuralNet *nn, Tensor *gradient, const Tensor *output,
                        const Tensor *target);

// Training
void nn_zero_gradients(NeuralNet *nn);
void nn_update_weights(NeuralNet *nn);
void nn_scale_gradients(NeuralNet *nn, float scale);
void nn_add_l2_gradient(NeuralNet *nn, float lambda);

// Testing
void test_nn_on_dataset(NeuralNet *nn, Dataset *data, const char *name);

#endif /* ifndef NN_H */
