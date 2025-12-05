/*
 * perceptron_examples.c - Perceptron example demonstrations
 */

#include "../activations/activations.h"
#include "../data/dataset.h"
#include "../nn/perceptron.h"
#include "../training/gradient_descent.h"
#include <stdio.h>

static int logic_gate_classifier(float prediction) {
    if (prediction <= 0.05) {
        return 0;
    }
    if (prediction >= 0.95) {
        return 1;
    }
    return -1;
}

void perceptron_learning_logic_gates() {
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
