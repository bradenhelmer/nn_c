/*
 * nn_internal.h - Internal NeuralNet struct definiton.
 */

#ifndef NN_INTERNAL_H
#define NN_INTERNAL_H
#include "nn.h"

struct NeuralNet {
    Layer **layers;
    int num_layers;
    LossType loss_type;
    Classifier classifier;
};

#endif /* ifndef NN_INTERNAL_H */
