/*
 * mnist_examples.c - MNSIT example demonstrations
 *
 */

#include "../activations/activations.h"
#include "../data/dataset.h"
#include "../training/gradient_descent.h"
#include "nn/nn.h"
#include <stdio.h>

static void mnist_classifier(Tensor *dest, const Tensor *prediction) {
    tensor_fill(dest, 0.0f);
    int max_index = tensor_argmax(prediction);
    dest->data[max_index] = 1.0f;
}

static void test_mnist(NeuralNet *nn, Dataset *data, const char *name) {
    printf("\nTesting %s:\n\n", name);
    int correct = 0;
    Tensor *input = tensor_create1d(data->X->shape[1]);
    Tensor *target = tensor_create1d(data->Y->shape[1]);
    Tensor *classification = tensor_create1d(data->Y->shape[1]);

    for (int i = 0; i < data->num_samples; i++) {
        tensor_get_row(input, data->X, i);
        tensor_get_row(target, data->Y, i);
        Tensor *prediction = nn_forward(nn, input);
        nn->classifier(classification, prediction);

        if (tensor_equals(classification, target)) {
            correct++;
        }
    }

    tensor_free(input);
    tensor_free(target);
    tensor_free(classification);

    printf("Final images correctly classified: %d\n", correct);
}

void mnist_sgd() {
    printf("\n\nTraining MNIST with SGD optimizer...\n");

    Dataset *mnist_train = create_mnist_train_dataset();
    Dataset *mnist_test = create_mnist_test_dataset();

    TrainingConfig config = {.max_epochs = 10,
                             .tolerance = 1e-7,
                             .batch_size = 64,
                             .verbose = 1,
                             .optimizer = optimizer_create_sgd(0.1f)};

    NeuralNet *nn_mnist = nn_create(4, 0.5f, TENSOR_SOFTMAX_CROSS_ENTROPY_LOSS, mnist_classifier);
    nn_add_layer(nn_mnist, 0, linear_layer_create(784, 128));
    nn_add_layer(nn_mnist, 1, activation_layer_create(TENSOR_RELU_ACTIVATION));
    nn_add_layer(nn_mnist, 2, linear_layer_create(128, 10));
    nn_add_layer(nn_mnist, 3, activation_layer_create(TENSOR_LINEAR_ACTIVATION));
    optimizer_init(config.optimizer, nn_mnist);

    TrainingResult *mnist_sgd_result = train_nn_batch_opt(nn_mnist, mnist_train, NULL, &config);

    printf("\nMNIST Batched Training with SGD Optimizer stopped at %d epochs\n",
           mnist_sgd_result->epochs_completed);
    printf("Final loss: %.6f\n", mnist_sgd_result->final_loss);
    printf("Final accuracy: %.2f%%\n",
           mnist_sgd_result->accuracy_history[mnist_sgd_result->epochs_completed - 1] * 100);

    test_mnist(nn_mnist, mnist_test, "MNIST test images on batched SGD-trained 2-layer NN");

    nn_free(nn_mnist);
    optimizer_free(config.optimizer);
    dataset_free(mnist_train);
    dataset_free(mnist_test);
    training_result_free(mnist_sgd_result);
}

void mnist_momentum() {
    printf("\n\nTraining MNIST with momentum optimizer...\n");

    Dataset *mnist_train = create_mnist_train_dataset();
    Dataset *mnist_test = create_mnist_test_dataset();

    TrainingConfig config = {.max_epochs = 10,
                             .tolerance = 1e-7,
                             .batch_size = 64,
                             .verbose = 1,
                             .optimizer = optimizer_create_momentum(0.01f, 0.9f)};

    NeuralNet *nn_mnist = nn_create(4, 0.5f, TENSOR_SOFTMAX_CROSS_ENTROPY_LOSS, mnist_classifier);
    nn_add_layer(nn_mnist, 0, linear_layer_create(784, 128));
    nn_add_layer(nn_mnist, 1, activation_layer_create(TENSOR_RELU_ACTIVATION));
    nn_add_layer(nn_mnist, 2, linear_layer_create(128, 10));
    nn_add_layer(nn_mnist, 3, activation_layer_create(TENSOR_LINEAR_ACTIVATION));
    optimizer_init(config.optimizer, nn_mnist);

    TrainingResult *mnist_sgd_result = train_nn_batch_opt(nn_mnist, mnist_train, NULL, &config);

    printf("\nMNIST Batched Training with momentum optimizer stopped at %d epochs\n",
           mnist_sgd_result->epochs_completed);
    printf("Final loss: %.6f\n", mnist_sgd_result->final_loss);
    printf("Final accuracy: %.2f%%\n",
           mnist_sgd_result->accuracy_history[mnist_sgd_result->epochs_completed - 1] * 100);

    test_mnist(nn_mnist, mnist_test, "MNIST test images on batched momentum-trained 2-layer NN");

    nn_free(nn_mnist);
    optimizer_free(config.optimizer);
    dataset_free(mnist_train);
    dataset_free(mnist_test);
    training_result_free(mnist_sgd_result);
}

void mnist_adam() {
    printf("\n\nTraining MNIST with ADAM optimizer...\n");

    Dataset *mnist_train = create_mnist_train_dataset();
    Dataset *mnist_test = create_mnist_test_dataset();

    TrainingConfig config = {.max_epochs = 10,
                             .batch_size = 64,
                             .verbose = 1,
                             .optimizer = optimizer_create_adam(0.001f, 0.9f, 0.999f, 1e-8)};

    NeuralNet *nn_mnist = nn_create(4, 0.5f, TENSOR_SOFTMAX_CROSS_ENTROPY_LOSS, mnist_classifier);
    nn_add_layer(nn_mnist, 0, linear_layer_create(784, 128));
    nn_add_layer(nn_mnist, 1, activation_layer_create(TENSOR_RELU_ACTIVATION));
    nn_add_layer(nn_mnist, 2, linear_layer_create(128, 10));
    nn_add_layer(nn_mnist, 3, activation_layer_create(TENSOR_LINEAR_ACTIVATION));
    optimizer_init(config.optimizer, nn_mnist);

    TrainingResult *mnist_sgd_result = train_nn_batch_opt(nn_mnist, mnist_train, NULL, &config);

    printf("\nMNIST Batched Training with ADAM optimizer stopped at %d epochs\n",
           mnist_sgd_result->epochs_completed);
    printf("Final loss: %.6f\n", mnist_sgd_result->final_loss);
    printf("Final accuracy: %.2f%%\n",
           mnist_sgd_result->accuracy_history[mnist_sgd_result->epochs_completed - 1] * 100);

    test_mnist(nn_mnist, mnist_test, "MNIST test images on batched ADAM-trained 2-layer NN");

    nn_free(nn_mnist);
    optimizer_free(config.optimizer);
    dataset_free(mnist_train);
    dataset_free(mnist_test);
    training_result_free(mnist_sgd_result);
}

void mnist_aggressive() {
    printf(
        "\n\nTraining MNIST with ADAM optimizer/cosine annealing scheduler/L2 regularization...\n");

    Dataset *mnist_train = create_mnist_train_dataset();
    Dataset *mnist_test = create_mnist_test_dataset();

    TrainingConfig config = {.max_epochs = 10,
                             .batch_size = 64,
                             .verbose = 1,
                             .optimizer = optimizer_create_adam(0.001f, 0.9f, 0.999f, 1e-8),
                             .scheduler = scheduler_create_cosine(0.001f, 1e-5f, 20),
                             .l2_lambda = 1e-4f};

    NeuralNet *nn_mnist = nn_create(4, 0.5f, TENSOR_SOFTMAX_CROSS_ENTROPY_LOSS, mnist_classifier);
    nn_add_layer(nn_mnist, 0, linear_layer_create(784, 128));
    nn_add_layer(nn_mnist, 1, activation_layer_create(TENSOR_RELU_ACTIVATION));
    nn_add_layer(nn_mnist, 2, linear_layer_create(128, 10));
    nn_add_layer(nn_mnist, 3, activation_layer_create(TENSOR_LINEAR_ACTIVATION));
    optimizer_init(config.optimizer, nn_mnist);

    TrainingResult *mnist_sgd_result = train_nn_batch_opt(nn_mnist, mnist_train, NULL, &config);

    printf("\nMNIST Batched Training with ADAM/cosine annealing stopped at %d epochs\n",
           mnist_sgd_result->epochs_completed);
    printf("Final loss: %.6f\n", mnist_sgd_result->final_loss);
    printf("Final accuracy: %.2f%%\n",
           mnist_sgd_result->accuracy_history[mnist_sgd_result->epochs_completed - 1] * 100);

    test_mnist(nn_mnist, mnist_test,
               "MNIST test images on batched ADAM-trained 2-layer NN with cosine annealing and L2 "
               "regularization");

    nn_free(nn_mnist);
    optimizer_free(config.optimizer);
    scheduler_free(config.scheduler);
    dataset_free(mnist_train);
    dataset_free(mnist_test);
    training_result_free(mnist_sgd_result);
}
