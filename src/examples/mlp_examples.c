/*
 * mlp_examples.c - Multi-layer perceptron example demonstrations
 */

#include "../activations/activations.h"
#include "../data/dataset.h"
#include "../nn/mlp.h"
#include "../training/gradient_descent.h"
#include <stdio.h>

static void xor_classifier(Tensor *dest, const Tensor *prediction) {
    if (prediction->data[0] <= 0.05) {
        dest->data[0] = 0.0f;
        return;
    }
    if (prediction->data[0] >= 0.95) {
        dest->data[0] = 1.0f;
        return;
    }
    dest->data[0] = -1.f;
}

void mlp_learning_xor() {

    printf("\n\nTraining XOR Gate with MLP...\n");

    TrainingConfig config = {.max_epochs = 20000, .tolerance = 1e-7, .batch_size = 1, .verbose = 0};
    Dataset *xor_data = create_xor_gate_dataset();

    MLP *mlp_xor = mlp_create(2, 0.5f, TENSOR_MSE_LOSS, xor_classifier);
    LinearLayer *layer_1 = linear_layer_create(2, 2, TENSOR_SIGMOID_ACTIVATION);
    linear_layer_init_xavier(layer_1);
    LinearLayer *layer_2 = linear_layer_create(2, 1, TENSOR_SIGMOID_ACTIVATION);
    linear_layer_init_xavier(layer_2);
    mlp_add_layer(mlp_xor, 0, layer_1);
    mlp_add_layer(mlp_xor, 1, layer_2);

    TrainingResult *result_xor = train_mlp(mlp_xor, xor_data, NULL, &config);

    printf("\nXOR Gate Training stopped at %d epochs\n", result_xor->epochs_completed);
    printf("Final loss: %.6f\n", result_xor->final_loss);
    printf("Final accuracy: %.2f%%\n",
           result_xor->accuracy_history[result_xor->epochs_completed - 1] * 100);

    test_mlp_on_dataset(mlp_xor, xor_data, "XOR Gate");

    mlp_free(mlp_xor);
    dataset_free(xor_data);
    training_result_free(result_xor);
}

void mlp_learning_xor_batched() {

    printf("\n\nTraining XOR Gate with Batched MLP...\n");

    TrainingConfig config = {.max_epochs = 50000, .tolerance = 1e-8, .batch_size = 1, .verbose = 0};
    Dataset *xor_data = create_xor_gate_dataset();

    MLP *mlp_xor = mlp_create(2, 0.3f, TENSOR_MSE_LOSS, xor_classifier);
    LinearLayer *layer_1 = linear_layer_create(2, 2, TENSOR_SIGMOID_ACTIVATION);
    linear_layer_init_xavier(layer_1);
    LinearLayer *layer_2 = linear_layer_create(2, 1, TENSOR_SIGMOID_ACTIVATION);
    linear_layer_init_xavier(layer_2);
    mlp_add_layer(mlp_xor, 0, layer_1);
    mlp_add_layer(mlp_xor, 1, layer_2);

    TrainingResult *result_xor = train_mlp_batch(mlp_xor, xor_data, NULL, &config);

    printf("\nBatched XOR Gate Training stopped at %d epochs\n", result_xor->epochs_completed);
    printf("Final loss: %.6f\n", result_xor->final_loss);
    printf("Final accuracy: %.2f%%\n",
           result_xor->accuracy_history[result_xor->epochs_completed - 1] * 100);

    test_mlp_on_dataset(mlp_xor, xor_data, "XOR Gate");

    mlp_free(mlp_xor);
    dataset_free(xor_data);
    training_result_free(result_xor);
}
