/*
 * mnist_gpu_examples.c - MNSIT GPU example demonstrations
 *
 */
#include "../activations/activations.h"
#include "../data/dataset.h"
#include "gpu/gpu_gradient_descent.h"
#include "gpu/gpu_nn.h"
#include "gpu/gpu_optimizer.h"
#include "nn/layer.h"
#include "nn/nn.h"
#include "utils/timing.h"
#include <stdio.h>

extern void test_mnist(NeuralNet *nn, Dataset *data, const char *name);
extern void mnist_classifier(Tensor *dest, const Tensor *prediction);

void mnist_gpu_basic() {
    printf("\n\nTraining MNIST with linear layer GPU execution...\n");

    Dataset *mnist_train = create_mnist_train_dataset();
    Dataset *mnist_test = create_mnist_test_dataset();

    TrainingConfig config = {.max_epochs = 1,
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

    // For fully-connected networks: height=1, width=1, channels=num_features
    InputShape input_shape = {.height = 1, .width = 1, .channels = 784};
    GPUNeuralNet *mnist_gpu_nn =
        gpu_nn_create_from_cpu_nn(nn_mnist, config.batch_size, input_shape);

    // Create and initialize GPU optimizer
    GPUOptimizer *gpu_opt = gpu_optimizer_create_sgd(0.1f);
    gpu_optimizer_init(gpu_opt, mnist_gpu_nn);

    Timer training_timer = {0};
    timer_start(&training_timer);
    TrainingResult *mnist_gpu_basic_result =
        train_nn_gpu_batch(mnist_gpu_nn, gpu_opt, mnist_train, NULL, &config);
    timer_stop(&training_timer);
    printf("GPU Training took: %.3f seconds\n", training_timer.elapsed);

    printf("\nMNIST Batched Training with SGD Optimizer stopped at %d epochs\n",
           mnist_gpu_basic_result->epochs_completed);
    printf("Final loss: %.6f\n", mnist_gpu_basic_result->final_loss);
    printf("Final accuracy: %.2f%%\n",
           mnist_gpu_basic_result->accuracy_history[mnist_gpu_basic_result->epochs_completed - 1] *
               100);

    nn_free(nn_mnist);
    gpu_nn_free(mnist_gpu_nn);
    gpu_optimizer_free(gpu_opt);
    optimizer_free(config.optimizer);
    dataset_free(mnist_train);
    dataset_free(mnist_test);
    training_result_free(mnist_gpu_basic_result);
}

// void mnist_conv_gpu() {
//     printf("\n\nTraining MNIST with 2D convolutional Neural network on GPU...\n");
//     _print_conv_arch();
//
//     Dataset *mnist_train = create_mnist_train_dataset();
//     Dataset *mnist_test = create_mnist_test_dataset();
//
//     TrainingConfig config = {.max_epochs = PROFILING ? 1 : 10,
//                              .batch_size = 64,
//                              .verbose = 1,
//                              .optimizer = optimizer_create_adam(0.001f, 0.9f, 0.999f, 1e-8),
//                              .scheduler = scheduler_create_cosine(0.001f, 1e-5f, 20),
//                              .l2_lambda = 1e-4f};
//     NeuralNet *mnist_conv = nn_create(7, 0.5f, TENSOR_SOFTMAX_CROSS_ENTROPY_LOSS,
//     mnist_classifier); nn_add_layer(mnist_conv, 0, conv_layer_create(1, 32, 5, 1, 2));
//     nn_add_layer(mnist_conv, 1, activation_layer_create(TENSOR_RELU_ACTIVATION));
//     nn_add_layer(mnist_conv, 2, maxpool_layer_create(2, 2));
//     nn_add_layer(mnist_conv, 3, flatten_layer_create());
//     nn_add_layer(mnist_conv, 4, linear_layer_create(6272, 128));
//     nn_add_layer(mnist_conv, 5, activation_layer_create(TENSOR_RELU_ACTIVATION));
//     nn_add_layer(mnist_conv, 6, linear_layer_create(128, 10));
//     optimizer_init(config.optimizer, mnist_conv);
//
//     // For CNN: actual spatial dimensions of MNIST images
//     InputShape conv_input = {.height = 28, .width = 28, .channels = 1};
//     GPUNeuralNet *mnist_conv_gpu = gpu_nn_create_from_cpu_nn(mnist_conv, config.batch_size,
//     conv_input); printf("Workspace size: %zu\n", mnist_conv_gpu->workspace_size);
//
//     Timer training_timer = {0};
//     timer_start(&training_timer);
//     TrainingResult *mnist_conv_result =
//         train_nn_gpu_batch(mnist_conv_gpu, mnist_train, NULL, &config);
//     timer_stop(&training_timer);
//     printf("Training took: %.3f seconds\n", training_timer.elapsed);
//
//     //
//     //     printf(
//     //         "\nMNIST Batched Convolutional NN training with ADAM/cosine annealing stopped at
//     %d
//     //         epochs", mnist_conv_result->epochs_completed);
//     //     printf("\nFinal accuracy: %.2f%%\n",
//     //            mnist_conv_result->accuracy_history[mnist_conv_result->epochs_completed - 1] *
//     //            100);
//     //
//     // #if !PROFILING
//     //     test_mnist_conv(mnist_conv, mnist_test,
//     //                     "MNIST test images on batched convolutional NN with cosine annealing
//     and
//     //                     L2 " "regularization");
//     // #endif
//
//     gpu_nn_free(mnist_conv_gpu);
//     nn_free(mnist_conv);
//     optimizer_free(config.optimizer);
//     scheduler_free(config.scheduler);
//     dataset_free(mnist_train);
//     dataset_free(mnist_test);
//     // training_result_free(mnist_conv_result);
// }
