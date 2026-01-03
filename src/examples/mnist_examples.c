/*
 * mnist_examples.c - MNSIT example demonstrations
 *
 */
#include "../activations/activations.h"
#include "../data/dataset.h"
#include "../training/gradient_descent.h"
#include "../utils/timing.h"
#include "config.h"
#include "nn/layer.h"
#include "nn/loss.h"
#include "nn/nn.h"
#include <stdio.h>

void mnist_classifier(Tensor *dest, const Tensor *prediction) {
    tensor_fill(dest, 0.0f);
    int max_index = tensor_argmax(prediction);
    dest->data[max_index] = 1.0f;
}

void test_mnist(NeuralNet *nn, Dataset *data, const char *name) {
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

static void test_mnist_conv(NeuralNet *nn, Dataset *data, const char *name) {
    printf("\nTesting %s:\n\n", name);
    int correct = 0;
    Tensor *input = tensor_create1d(data->X->shape[1]);
    Tensor *target = tensor_create1d(data->Y->shape[1]);
    Tensor *classification = tensor_create1d(data->Y->shape[1]);

    for (int i = 0; i < data->num_samples; i++) {
        tensor_get_row(input, data->X, i);
        tensor_get_row(target, data->Y, i);
        Tensor *spatial_input = tensor_unflatten(input, 3, (int[]){1, 28, 28});
        Tensor *prediction = nn_forward(nn, spatial_input);
        nn->classifier(classification, prediction);

        if (tensor_equals(classification, target)) {
            correct++;
        }
        tensor_free(spatial_input);
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

    NeuralNet *nn_mnist = nn_create(4, 0.5f, LOSS_SOFTMAX_CROSS_ENTROPY, mnist_classifier);
    nn_add_layer(nn_mnist, 0, linear_layer_create(784, 128));
    nn_add_layer(nn_mnist, 1, activation_layer_create(ACTIVATION_RELU));
    nn_add_layer(nn_mnist, 2, linear_layer_create(128, 10));
    nn_add_layer(nn_mnist, 3, activation_layer_create(ACTIVATION_LINEAR));
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

    NeuralNet *nn_mnist = nn_create(4, 0.5f, LOSS_SOFTMAX_CROSS_ENTROPY, mnist_classifier);
    nn_add_layer(nn_mnist, 0, linear_layer_create(784, 128));
    nn_add_layer(nn_mnist, 1, activation_layer_create(ACTIVATION_RELU));
    nn_add_layer(nn_mnist, 2, linear_layer_create(128, 10));
    nn_add_layer(nn_mnist, 3, activation_layer_create(ACTIVATION_LINEAR));
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

    NeuralNet *nn_mnist = nn_create(4, 0.5f, LOSS_SOFTMAX_CROSS_ENTROPY, mnist_classifier);
    nn_add_layer(nn_mnist, 0, linear_layer_create(784, 128));
    nn_add_layer(nn_mnist, 1, activation_layer_create(ACTIVATION_RELU));
    nn_add_layer(nn_mnist, 2, linear_layer_create(128, 10));
    nn_add_layer(nn_mnist, 3, activation_layer_create(ACTIVATION_LINEAR));
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

    NeuralNet *nn_mnist = nn_create(4, 0.5f, LOSS_SOFTMAX_CROSS_ENTROPY, mnist_classifier);
    nn_add_layer(nn_mnist, 0, linear_layer_create(784, 128));
    nn_add_layer(nn_mnist, 1, activation_layer_create(ACTIVATION_RELU));
    nn_add_layer(nn_mnist, 2, linear_layer_create(128, 10));
    nn_add_layer(nn_mnist, 3, activation_layer_create(ACTIVATION_LINEAR));
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

static void _print_conv_arch() {
    printf("\tConv(1, 32, 5, 1, 2)   → 28×28×32\n");
    printf("\tReLU                   → 28×28×32\n");
    printf("\tMaxPool(2, 2)          → 14×14×32\n");
    printf("\tFlatten                → 6272\n");
    printf("\tLinear(6272, 128)      → 128\n");
    printf("\tReLU                   → 128\n");
    printf("\tLinear(128, 10)        → 10\n");
    printf("\tSoftmax (via loss)     → 10\n\n");
}

void mnist_conv() {
    printf("\n\nTraining MNIST with 2D convolutional Neural network...\n");
    _print_conv_arch();

    Dataset *mnist_train = create_mnist_train_dataset();
    Dataset *mnist_test = create_mnist_test_dataset();

    TrainingConfig config = {.max_epochs = PROFILING ? 1 : 10,
                             .batch_size = 64,
                             .verbose = 1,
                             .optimizer = optimizer_create_adam(0.001f, 0.9f, 0.999f, 1e-8),
                             .scheduler = scheduler_create_cosine(0.001f, 1e-5f, 20),
                             .l2_lambda = 1e-4f};
    NeuralNet *mnist_conv = nn_create(7, 0.5f, LOSS_SOFTMAX_CROSS_ENTROPY, mnist_classifier);
    nn_add_layer(mnist_conv, 0, conv2d_layer_create(1, 32, 5, 1, 2));
    nn_add_layer(mnist_conv, 1, activation_layer_create(ACTIVATION_RELU));
    nn_add_layer(mnist_conv, 2, maxpool_layer_create(2, 2));
    nn_add_layer(mnist_conv, 3, flatten_layer_create());
    nn_add_layer(mnist_conv, 4, linear_layer_create(6272, 128));
    nn_add_layer(mnist_conv, 5, activation_layer_create(ACTIVATION_RELU));
    nn_add_layer(mnist_conv, 6, linear_layer_create(128, 10));
    optimizer_init(config.optimizer, mnist_conv);

    Timer training_timer = {0};
    timer_start(&training_timer);
    TrainingResult *mnist_conv_result = train_nn_batch_opt(mnist_conv, mnist_train, NULL, &config);
    timer_stop(&training_timer);
    printf("Training took: %.3f seconds\n", training_timer.elapsed);

    printf(
        "\nMNIST Batched Convolutional NN training with ADAM/cosine annealing stopped at %d epochs",
        mnist_conv_result->epochs_completed);
    printf("\nFinal accuracy: %.2f%%\n",
           mnist_conv_result->accuracy_history[mnist_conv_result->epochs_completed - 1] * 100);

#if !PROFILING
    test_mnist_conv(mnist_conv, mnist_test,
                    "MNIST test images on batched convolutional NN with cosine annealing and L2 "
                    "regularization");
#endif

    nn_free(mnist_conv);
    optimizer_free(config.optimizer);
    scheduler_free(config.scheduler);
    dataset_free(mnist_train);
    dataset_free(mnist_test);
    training_result_free(mnist_conv_result);
}
