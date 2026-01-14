/*
 * dataset.h
 *
 * Struct and function declarations for dataset handling.
 */
#ifndef DATASET_H
#define DATASET_H

#include "core/tensor.h"

// Opaque type - see dataset_internal.h for struct definition
typedef struct Dataset Dataset;

Dataset *dataset_create(int num_samples, int num_features);
void dataset_free(Dataset *d);

// Accessors
int dataset_get_num_samples(const Dataset *d);
int dataset_get_num_features(const Dataset *d);
Tensor *dataset_get_X(const Dataset *d);
Tensor *dataset_get_Y(const Dataset *d);

// Logic Gate datasets
Dataset *dataset_create_and_gate();
Dataset *dataset_create_or_gate();
Dataset *dataset_create_xor_gate();

// MNIST Dataset
#define MNIST_IMG_PIXEL_COUNT 784
#define MNIST_OUTPUT_SIZE 10
#define MNIST_LOADER_BATCH_SIZE 128

#define MNIST_TRAIN_IMG_COUNT 60000
#define MNIST_TRAIN_IMG_PATH "datasets/mnist/train_imgs"
#define MNIST_TRAIN_LABEL_PATH "datasets/mnist/train_labels"

#define MNIST_TEST_IMG_COUNT 10000
#define MNIST_TEST_IMG_PATH "datasets/mnist/test_imgs"
#define MNIST_TEST_LABEL_PATH "datasets/mnist/test_labels"

Dataset *dataset_create_mnist_train();
Dataset *dataset_create_mnist_test();

#endif /* ifndef DATASET_H */
