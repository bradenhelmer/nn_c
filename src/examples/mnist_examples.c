/*
 * mnist_examples.c - MNSIT example demonstrations
 *
 */

#include "../activations/activations.h"
#include "../data/dataset.h"
#include "../linalg/vector.h"
#include "../nn/mlp.h"
#include "../training/gradient_descent.h"
#include <stdio.h>

static Vector *mnist_classifier(Vector *prediction) {
    float max = 0.f, curr;
    int max_index = 0;
    for (int i = 0; i < prediction->size; i++) {
        curr = prediction->data[i];
        if (curr > max) {
            max = curr;
            max_index = i;
        }
    }
    Vector *ret = vector_create(10);
    ret->data[max_index] = 1.0f;
    return ret;
}

void mnist_sgd() {
    printf("\n\nTraining MNIST with SGD...\n");

    Dataset *mnist_train = create_mnist_train_dataset();
    // Dataset *mnist_test = create_mnist_test_dataset();

    TrainingConfig config = {.max_epochs = 10,
                             .tolerance = 1e-7,
                             .batch_size = 32,
                             .verbose = 1,
                             .optimizer = optimizer_create_sgd(0.1f)};

    MLP *mlp_mnist = mlp_create(2, 0.5f, VECTOR_SOFTMAX_CROSS_ENTROPY_LOSS, mnist_classifier);
    Layer *layer_1 = layer_create(784, 128, VECTOR_RELU_ACTIVATION);
    Layer *layer_2 = layer_create(128, 10, VECTOR_LINEAR_ACTIVATION);
    mlp_add_layer(mlp_mnist, 0, layer_1);
    mlp_add_layer(mlp_mnist, 1, layer_2);
    optimizer_init(config.optimizer, mlp_mnist);

    TrainingResult *mnist_sgd_result = train_mlp_batch_opt(mlp_mnist, mnist_train, NULL, &config);

    printf("\nMNIST Batched Training with SGD Optimizer stopped at %d epochs\n",
           mnist_sgd_result->epochs_completed);
    printf("Final loss: %.6f\n", mnist_sgd_result->final_loss);
    printf("Final accuracy: %.2f%%\n",
           mnist_sgd_result->accuracy_history[mnist_sgd_result->epochs_completed - 1] * 100);
}

void mnist_momentum() {
}

void mnist_adam() {
}
