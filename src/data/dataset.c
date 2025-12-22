/*
 * dataset.c
 *
 * Dataset function implementations.
 */
#include "dataset.h"
#include <stdio.h>
#include <stdlib.h>

Dataset *dataset_create(int num_samples, int num_features) {
    Dataset *d = (Dataset *)malloc(sizeof(Dataset));
    d->num_samples = num_samples;
    d->num_features = num_features;
    return d;
}

void dataset_free(Dataset *d) {
    tensor_free(d->X);
    tensor_free(d->Y);
    free(d);
}

static Tensor *_create_2bit_input_tensor() {
    Tensor *t = tensor_create2d(4, 2);
    tensor_set2d(t, 1, 1, 1);
    tensor_set2d(t, 2, 0, 1);
    tensor_set2d(t, 3, 0, 1);
    tensor_set2d(t, 3, 1, 1);
    return t;
}

// Test datasets
Dataset *create_and_gate_dataset() {
    Dataset *d = dataset_create(4, 2);
    d->X = _create_2bit_input_tensor();
    d->Y = tensor_create2d(4, 1);
    tensor_set2d(d->Y, 3, 0, 1);
    return d;
}

Dataset *create_or_gate_dataset() {
    Dataset *d = dataset_create(4, 2);
    d->X = _create_2bit_input_tensor();
    d->Y = tensor_create2d(4, 1);
    tensor_fill(d->Y, 1.0f);
    tensor_set2d(d->Y, 0, 0, 0);
    return d;
}

Dataset *create_xor_gate_dataset() {
    Dataset *d = dataset_create(4, 2);
    d->X = _create_2bit_input_tensor();
    d->Y = tensor_create2d(4, 1);
    tensor_set2d(d->Y, 1, 0, 1);
    tensor_set2d(d->Y, 2, 0, 1);
    return d;
}

/*
 * Batched MNIST file loader.
 *
 * Visit https://yann.lecun.org/exdb/mnist/index.html for file formats.
 */
static Dataset *load_mnist_file(const char *image_path, const char *label_path,
                                const int image_count) {

    // Create initial train dataset and X tensor (image_count x 784 pixels)
    Dataset *d = dataset_create(image_count, MNIST_IMG_PIXEL_COUNT);
    d->X = tensor_create2d(image_count, MNIST_IMG_PIXEL_COUNT);

    // Open file, seek past magic number (16 bytes).
    FILE *train_img_file = fopen(image_path, "rb");
    if (!train_img_file) {
        fprintf(stderr, "Error: Could not open image file: %s\n", image_path);
        exit(1);
    }
    fseek(train_img_file, 16, SEEK_CUR);

    unsigned char *img_batch = (unsigned char *)malloc(
        sizeof(unsigned char) * MNIST_IMG_PIXEL_COUNT * MNIST_LOADER_BATCH_SIZE);

    int num_batches = image_count / MNIST_LOADER_BATCH_SIZE;
    int remainder = image_count % MNIST_LOADER_BATCH_SIZE;

    for (int batch = 0; batch < num_batches; batch++) {
        fread(img_batch, sizeof(unsigned char), MNIST_IMG_PIXEL_COUNT * MNIST_LOADER_BATCH_SIZE,
              train_img_file);

        for (int i = 0; i < MNIST_LOADER_BATCH_SIZE; i++) {
            for (int j = 0; j < MNIST_IMG_PIXEL_COUNT; j++) {
                d->X->data[(batch * MNIST_LOADER_BATCH_SIZE + i) * MNIST_IMG_PIXEL_COUNT + j] =
                    img_batch[i * MNIST_IMG_PIXEL_COUNT + j] / 255.0f;
            }
        }
    }

    // Load remaining images that don't fill a complete batch
    if (remainder > 0) {
        fread(img_batch, sizeof(unsigned char), MNIST_IMG_PIXEL_COUNT * remainder, train_img_file);

        int base_idx = num_batches * MNIST_LOADER_BATCH_SIZE;
        for (int i = 0; i < remainder; i++) {
            for (int j = 0; j < MNIST_IMG_PIXEL_COUNT; j++) {
                d->X->data[(base_idx + i) * MNIST_IMG_PIXEL_COUNT + j] =
                    img_batch[i * MNIST_IMG_PIXEL_COUNT + j] / 255.0f;
            }
        }
    }

    free(img_batch);

    // Create Y (label tensor) (image_count x 10 outputs)
    d->Y = tensor_create2d(image_count, MNIST_OUTPUT_SIZE);

    // Open file, seek past magic number (8 bytes).
    FILE *train_label_file = fopen(label_path, "rb");
    if (!train_label_file) {
        fprintf(stderr, "Error: Could not open label file: %s\n", label_path);
        exit(1);
    }
    fseek(train_label_file, 8, SEEK_CUR);

    // Read all bytes into conversion buffer.
    unsigned char *label_conversion_buffer =
        (unsigned char *)malloc(sizeof(unsigned char) * image_count);
    fread(label_conversion_buffer, sizeof(char), image_count, train_label_file);
    for (int i = 0; i < image_count; i++) {
        // Since label in [0, 9], y[row][label] = 1.0
        d->Y->data[i * MNIST_OUTPUT_SIZE + label_conversion_buffer[i]] = 1.0f;
    }
    free(label_conversion_buffer);

    // Close files
    fclose(train_img_file);
    fclose(train_label_file);
    return d;
}

Dataset *create_mnist_train_dataset() {
    return load_mnist_file(MNIST_TRAIN_IMG_PATH, MNIST_TRAIN_LABEL_PATH, MNIST_TRAIN_IMG_COUNT);
}

Dataset *create_mnist_test_dataset() {
    return load_mnist_file(MNIST_TEST_IMG_PATH, MNIST_TEST_LABEL_PATH, MNIST_TEST_IMG_COUNT);
}
