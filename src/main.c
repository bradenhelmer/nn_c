/*
 * main.c - Neural network application entry point
 */

#include "activations/activations.h"
#include "data/dataset.h"
#include "nn/mlp.h"
#include "nn/perceptron.h"
#include "training/gradient_descent.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static int logic_gate_classifier(float prediction) {
    if (prediction <= 0.05) {
        return 0;
    }
    if (prediction >= 0.95) {
        return 1;
    }
    return -1;
}

static void perceptron_learning_logic_gates() {
    printf("=== Perceptron Learning Logic Gates ===\n\n");

    // AND GATE
    printf("Training AND Gate...\n");
    Dataset *and_data = create_and_gate_dataset();

    Perceptron *p_and =
        perceptron_create(2, -1.f, 1.f, 1.0f, SIGMOID_ACTIVATION, logic_gate_classifier);

    TrainingConfig config = {.max_epochs = 20000, .tolerance = 1e-7, .batch_size = 1, .verbose = 0};

    TrainingResult *result_and = train_perceptron(p_and, and_data, NULL, &config);
    printf("AND gate training completed in %d epochs\n", result_and->epochs_completed);
    printf("Final loss: %.6f\n", result_and->final_loss);
    printf("Final accuracy: %.6f\n",
           result_and->accuracy_history[result_and->epochs_completed - 1] * 100);
    test_perceptron_on_dataset(p_and, and_data, "AND Gate");

    // OR GATE
    printf("\n\nTraining OR Gate...\n");
    Dataset *or_data = create_or_gate_dataset();

    Perceptron *p_or =
        perceptron_create(2, -1.f, 1.f, 1.0f, SIGMOID_ACTIVATION, logic_gate_classifier);

    TrainingResult *result_or = train_perceptron(p_or, or_data, NULL, &config);
    printf("OR gate training completed in %d epochs\n", result_or->epochs_completed);
    printf("Final loss: %.6f\n", result_or->final_loss);
    printf("Final accuracy: %.6f\n",
           result_or->accuracy_history[result_or->epochs_completed - 1] * 100);
    test_perceptron_on_dataset(p_or, or_data, "OR Gate");

    // XOR GATE
    printf("\n\nTraining XOR Gate (should fail to converge)...\n");
    Dataset *xor_data = create_xor_gate_dataset();

    Perceptron *p_xor =
        perceptron_create(2, -1.f, 1.f, 1.0f, SIGMOID_ACTIVATION, logic_gate_classifier);

    config.max_epochs = 100;

    TrainingResult *result_xor = train_perceptron(p_xor, xor_data, NULL, &config);

    printf("\nXOR Gate Training stopped at %d epochs\n", result_xor->epochs_completed);
    printf("Final loss: %.6f (should be high)\n", result_xor->final_loss);
    printf("Final accuracy: %.2f%% (should be ~0%%)\n",
           result_xor->accuracy_history[result_xor->epochs_completed - 1] * 100);

    test_perceptron_on_dataset(p_xor, xor_data, "XOR Gate");
    printf("\nNote: XOR is not linearly separable - single perceptron cannot learn it!\n");
    printf("This motivates Phase 3: Multi-layer perceptrons.\n");

    // Free all memory
    perceptron_free(p_and);
    perceptron_free(p_or);
    perceptron_free(p_xor);
    dataset_free(and_data);
    dataset_free(or_data);
    dataset_free(xor_data);
    training_result_free(result_and);
    training_result_free(result_or);
    training_result_free(result_xor);
}

static Vector *xor_classifier(Vector *prediction) {
    if (prediction->data[0] <= 0.05) {
        return vector_zeros(1);
    }
    if (prediction->data[0] >= 0.95) {
        return vector_ones(1);
    }
    Vector *wrong = vector_create(1);
    wrong->data[0] = -1.f;
    return wrong;
}

static void mlp_learning_xor() {

    printf("\n\nTraining XOR Gate with MLP...\n");

    TrainingConfig config = {.max_epochs = 20000, .tolerance = 1e-7, .batch_size = 1, .verbose = 0};
    Dataset *xor_data = create_xor_gate_dataset();

    MLP *mlp_xor = mlp_create(2, 0.5f, VECTOR_MSE_LOSS, xor_classifier);
    Layer *layer_1 = layer_create(2, 2, VECTOR_SIGMOID_ACTIVATION);
    layer_init_xavier(layer_1);
    Layer *layer_2 = layer_create(2, 1, VECTOR_SIGMOID_ACTIVATION);
    layer_init_xavier(layer_2);
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

static void mlp_learning_xor_batched() {

    printf("\n\nTraining XOR Gate with Batched MLP...\n");

    TrainingConfig config = {.max_epochs = 50000, .tolerance = 1e-8, .batch_size = 1, .verbose = 0};
    Dataset *xor_data = create_xor_gate_dataset();

    MLP *mlp_xor = mlp_create(2, 0.3f, VECTOR_MSE_LOSS, xor_classifier);
    Layer *layer_1 = layer_create(2, 2, VECTOR_SIGMOID_ACTIVATION);
    layer_init_xavier(layer_1);
    Layer *layer_2 = layer_create(2, 1, VECTOR_SIGMOID_ACTIVATION);
    layer_init_xavier(layer_2);
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

int main() {
    srand(time(NULL));
    // perceptron_learning_logic_gates();
    // mlp_learning_xor();
    mlp_learning_xor_batched();
    return 0;
}
