/*
 * dataset.h
 *
 * Struct and function declarations for dataset handling.
 */
#ifndef DATASET_H
#define DATASET_H
#include "../linalg/matrix.h"
#include "../linalg/vector.h"

typedef struct {
    Matrix *X;
    Matrix *Y;
    int num_samples;
    int num_features;
} Dataset;

Dataset *dataset_create(int num_samples, int num_features);
void dataset_free(Dataset *d);

// Logic Gate datasets
Dataset *create_and_gate_dataset();
Dataset *create_or_gate_dataset();
Dataset *create_xor_gate_dataset();

// MNIST Dataset
#define MNIST_IMG_PIXEL_COUNT 784
#define MNIST_OUTPUT_SIZE 10
#define MNIST_LOADER_BATCH_SIZE 128

#define MNIST_TRAIN_IMG_COUNT 60000
#define MNIST_TRAIN_IMG_PATH "src/data/dataset/mnist/train_imgs"
#define MNIST_TRAIN_LABEL_PATH "src/data/dataset/mnist/train_labels"

#define MNIST_TEST_IMG_COUNT 10000
#define MNIST_TEST_IMG_PATH "src/data/dataset/mnist/test_imgs"
#define MNIST_TEST_LABEL_PATH "src/data/dataset/mnist/test_labels"

Dataset *create_mnist_train_dataset();
Dataset *create_mnist_test_dataset();

#endif /* ifndef DATASET_H */
