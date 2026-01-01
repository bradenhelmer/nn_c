/*
 * gpu_gradient_descent.h
 *
 * GPU Gradient descent definitions
 */
#ifndef GPU_GRADIENT_DESCENT_H
#define GPU_GRADIENT_DESCENT_H
#include "data/dataset.h"
#include "gpu/gpu_nn.h"
#include "training/gradient_descent.h"

TrainingResult *train_nn_gpu_batch(GPUNeuralNet *gpu_nn, Dataset *train_data, Dataset *val_data,
                                   TrainingConfig *config);

#endif /* ifndef GPU_GRADIENT_DESCENT_H */
