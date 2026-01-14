/*
 * dataset_internal.h - Internal Dataset struct definition
 */

#ifndef DATASET_INTERNAL_H
#define DATASET_INTERNAL_H
#include "core/tensor.h"
#include "dataset.h"

struct Dataset {
    Tensor *X; // 2D: (num_samples, num_features)
    Tensor *Y; // 2D: (num_samples, num_outputs)
    int num_samples;
    int num_features;
};

#endif /* ifndef DATASET_INTERNAL_H */
