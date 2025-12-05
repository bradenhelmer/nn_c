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
    matrix_free(d->X);
    matrix_free(d->Y);
    free(d);
}

static Matrix *_create_2bit_input_matrix() {
    Matrix *m = matrix_zeros(4, 2);
    matrix_set(m, 1, 1, 1);
    matrix_set(m, 2, 0, 1);
    matrix_set(m, 3, 0, 1);
    matrix_set(m, 3, 1, 1);
    return m;
}

// Test datasets
Dataset *create_and_gate_dataset() {
    Dataset *d = dataset_create(4, 2);
    d->X = _create_2bit_input_matrix();
    d->Y = matrix_zeros(4, 1);
    matrix_set(d->Y, 3, 0, 1);
    return d;
}

Dataset *create_or_gate_dataset() {
    Dataset *d = dataset_create(4, 2);
    d->X = _create_2bit_input_matrix();
    d->Y = matrix_ones(4, 1);
    matrix_set(d->Y, 0, 0, 0);
    return d;
}

Dataset *create_xor_gate_dataset() {
    Dataset *d = dataset_create(4, 2);
    d->X = _create_2bit_input_matrix();
    d->Y = matrix_zeros(4, 1);
    matrix_set(d->Y, 1, 0, 1);
    matrix_set(d->Y, 2, 0, 1);
    return d;
}

/*
 * Batched MNIST file loader.
 *
 * Visit https://yann.lecun.org/exdb/mnist/index.html for file formats.
 */
static Dataset *load_mnist_file(const char *image_path, const char *label_path,
                                const int image_count) {

    // Create initial train dataset and X matrix (60,0000 x 784 pixels)
    Dataset *d = dataset_create(image_count, MNIST_IMG_PIXEL_COUNT);
    d->X = matrix_create(image_count, MNIST_IMG_PIXEL_COUNT);

    // Open file, seek past magic number (16 bytes).
    FILE *train_img_file = fopen(image_path, "rb");
    fseek(train_img_file, 16, SEEK_CUR);

    unsigned char *img_batch = (unsigned char *)malloc(
        sizeof(unsigned char) * MNIST_IMG_PIXEL_COUNT * MNIST_LOADER_BATCH_SIZE);

    for (int batch = 0; batch < image_count / MNIST_LOADER_BATCH_SIZE; batch++) {
        fread(img_batch, sizeof(unsigned char), MNIST_IMG_PIXEL_COUNT * MNIST_LOADER_BATCH_SIZE,
              train_img_file);

        for (int i = 0; i < MNIST_LOADER_BATCH_SIZE; i++) {
            for (int j = 0; j < MNIST_IMG_PIXEL_COUNT; j++) {
                d->X->data[(batch * MNIST_LOADER_BATCH_SIZE + i) * MNIST_IMG_PIXEL_COUNT + j] =
                    img_batch[i * MNIST_IMG_PIXEL_COUNT + j] / 255.0f;
            }
        }
    }

    // Create Y (label matrix) (60,000 img x 10 outputs)
    d->Y = matrix_create(image_count, MNIST_OUTPUT_SIZE);

    // Open file, seek past magic number (8 bytes).
    FILE *train_label_file = fopen(label_path, "rb");
    fseek(train_label_file, 8, SEEK_CUR);

    // Read all bytes into conversion buffer.
    unsigned char label_conversion_buffer[image_count];
    fread(label_conversion_buffer, sizeof(char), image_count, train_label_file);
    for (int i = 0; i < image_count; i++) {
        // Since label in [0, 9], y[row][label] = 1.0
        d->Y->data[i * MNIST_OUTPUT_SIZE + label_conversion_buffer[i]] = 1.0f;
    }

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
