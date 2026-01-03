/*
 * mlp_examples.c - Multi-layer perceptron example demonstrations
 */

#include "../activations/activations.h"
#include "../data/dataset.h"
#include "../nn/nn.h"
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

void nn_learning_xor() {

    printf("\nTraining XOR Gate with 2-layer NN...\n");

    TrainingConfig config = {.max_epochs = 10000, .tolerance = 1e-7, .batch_size = 1, .verbose = 0};
    Dataset *xor_data = create_xor_gate_dataset();

    NeuralNet *nn_xor = nn_create(4, 0.5f, LOSS_MSE, xor_classifier);
    nn_add_layer(nn_xor, 0, linear_layer_create(2, 2));
    nn_add_layer(nn_xor, 1, activation_layer_create(ACTIVATION_SIGMOID));
    nn_add_layer(nn_xor, 2, linear_layer_create(2, 1));
    nn_add_layer(nn_xor, 3, activation_layer_create(ACTIVATION_SIGMOID));

    TrainingResult *result_xor = train_nn(nn_xor, xor_data, NULL, &config);

    printf("\nXOR Gate Training stopped at %d epochs\n", result_xor->epochs_completed);
    printf("Final loss: %.6f\n", result_xor->final_loss);
    printf("Final accuracy: %.2f%%\n",
           result_xor->accuracy_history[result_xor->epochs_completed - 1] * 100);

    test_nn_on_dataset(nn_xor, xor_data, "XOR Gate");

    nn_free(nn_xor);
    dataset_free(xor_data);
    training_result_free(result_xor);
}

void nn_learning_xor_batched() {

    printf("\n\nTraining XOR Gate with Batched 2-layer NN...\n");

    TrainingConfig config = {.max_epochs = 50000, .tolerance = 1e-7, .batch_size = 1, .verbose = 0};
    Dataset *xor_data = create_xor_gate_dataset();

    NeuralNet *nn_xor = nn_create(4, 0.3f, LOSS_MSE, xor_classifier);
    nn_add_layer(nn_xor, 0, linear_layer_create(2, 2));
    nn_add_layer(nn_xor, 1, activation_layer_create(ACTIVATION_SIGMOID));
    nn_add_layer(nn_xor, 2, linear_layer_create(2, 1));
    nn_add_layer(nn_xor, 3, activation_layer_create(ACTIVATION_SIGMOID));

    TrainingResult *result_xor = train_nn_batch(nn_xor, xor_data, NULL, &config);

    printf("\nBatched XOR Gate Training stopped at %d epochs\n", result_xor->epochs_completed);
    printf("Final loss: %.6f\n", result_xor->final_loss);
    printf("Final accuracy: %.2f%%\n",
           result_xor->accuracy_history[result_xor->epochs_completed - 1] * 100);

    test_nn_on_dataset(nn_xor, xor_data, "XOR Gate");

    nn_free(nn_xor);
    dataset_free(xor_data);
    training_result_free(result_xor);
}
