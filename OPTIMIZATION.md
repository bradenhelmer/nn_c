# Optimizations

This is a timeline of the various optimizations made to speed up the training of my convolutional neural network on the MNIST dataset.

## Neural Net Architecture

The network trained on the MNIST composes of 7 layers:

1. 2D Convolutional Layer (Input Tensor (1x28x28))
    - Input channels: 1
    - Output channels: 32
    - Kernel size: 5x5
    - Stride: 1
    - Padding: 2

2. ReLU Layer

3. Max Pooling Layer:
    - Pool size: 2
    - Stride: 2

4. Flatten Layer

5. Linear Layer:
    - Input: 6272
    - Output: 128

6. ReLU Layer

7. Linear Layer:
    - Input: 128
    - Output: 10

Loss Function: Softmax Cross Entropy

## Base Run

### Training Configuration

The training code can be found in the `mnist_conv` function in [mnist_examples.c](./src/examples/mnist_examples.c)

- Method: Batched gradient-descent
- Optimzer: ADAM (learning_rate: 0.001, beta1: 0.9, beta2: 0.999, epsilon: 1e-8)
- Scheduler: Cosine Annealing (initial_learning_rate: 0.001, min_learning_rate: 1e-5, max_timestep: 20)
- Max Epochs: 10
- L2 Lambda regularization: 0.0001
- Learning Rate: 0.5

### Compiler Information

For a baseline profile, we can compile with flags for a release build:
```sh
clang -std=c99 -Wall -Wextra -O3 -march=native -flto -DNDEBUG ...
```

### Timer

We'll use a simple POSIX timer to time the baseline training loop:

```c
// timer.h
#define _POSIX_C_SOURCE 199309L
#include <time.h>

typedef struct {
    struct timespec start;
    double elapsed;
} Timer;

static inline void timer_start(Timer *t) {
    clock_gettime(CLOCK_MONOTONIC, &t->start);
}

static inline void timer_stop(Timer *t) {
    struct timespec end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    t->elapsed = (end.tv_sec - t->start.tv_sec) + (end.tv_nsec - t->start.tv_nsec) * 1e-9;
}
```

Integrated with the training logic:

```c
// mnist_examples.c
Timer training_timer = {0};
timer_start(&training_timer);
TrainingResult *mnist_conv_result = train_nn_batch_opt(mnist_conv, mnist_train, NULL, &config);
timer_stop(&training_timer);
printf("Training took: %f seconds\n", training_timer.elapsed);
```

### Baseline Results

```
Epoch 0: loss=0.2594, accuracy=92.58%
Epoch 1: loss=0.0983, accuracy=97.04%
Epoch 2: loss=0.0757, accuracy=97.64%
Epoch 3: loss=0.0674, accuracy=97.91%
Epoch 4: loss=0.0585, accuracy=98.21%
Epoch 5: loss=0.0477, accuracy=98.50%
Epoch 6: loss=0.0418, accuracy=98.71%
Epoch 7: loss=0.0412, accuracy=98.66%
Epoch 8: loss=0.0408, accuracy=98.71%
Epoch 9: loss=0.0331, accuracy=99.00%
Training took: 1556.741196 seconds

MNIST Batched Convolutional NN training with ADAM/cosine annealing stopped at 10 epochs
Final accuracy: 99.00%

Testing MNIST test images on batched convolutional NN with cosine annealing and L2 regularization:

Final images correctly classified: 9868
```

Some pretty promising results from a model perspective, but that was slow.
Just under 26 minutes total time, around 2 and a half minutes per epoch.

## Initial Profiling & Indexing Optimization

To analyze the run, lets create a build for running `perf` using the following flags:
```
PERF_FLAGS  := -O3 -march=native -flto -g -fno-omit-frame-pointer -DNDEBUG -DPROFILING=1
```
We add the `-fno-omit-frame-pointer` flag when compiling to force the compiler to reserve the `RBP` register for the stack pointer.
This allows tools like perf to walk the stack without DWARF info (we still add the `-g` flag for call graph recording).
Lastly, we'll define `PROFILING=1` and add some directives in the code to only run for one epoch and only 1000 samples:

```c
// gradient_descent.c
#if PROFILING
            if (samples_seen >= 1000) {
                break;
            }
#endif
```

```c
// mnist_examples.c
TrainingConfig config = {.max_epochs = PROFILING ? 1 : 10};
```

We'll also add some make targets for convenience:

```makefile
perf-record: perf-build
	@echo "$(YELLOW)Recording perf profile...$(RESET)"
	@sudo perf record -g -F 4000 --call-graph dwarf ./$(PERF_TARGET)
	@echo "$(GREEN)Profile saved to perf.data$(RESET)"

perf-report:
	@echo "$(YELLOW)Generating perf report...$(RESET)"
	@sudo perf report --stdio --no-children -n | head -100
```

Recording the run and viewing the report:

```
# Total Lost Samples: 0
#
# Samples: 10K of event 'cycles:P'
# Event count (approx.): 15064876132
#
# Overhead       Samples  Command          Shared Object      Symbol
# ........  ............  ...............  .................  .........................................
#
    94.40%         10247  neural_net_perf  neural_net_perf    [.] train_nn_batch_opt
            |
            |--41.78%--train_nn_batch_opt
            |          mnist_conv (inlined)
            |          main
            |          __libc_start_call_main
            |          __libc_start_main_impl (inlined)
            |          _start
            |
            |--34.05%--nn_forward (inlined)
            |          train_nn_batch_opt
            |          |
            |           --34.02%--mnist_conv (inlined)
            |                     main
            |                     __libc_start_call_main
            |                     __libc_start_main_impl (inlined)
            |                     _start
            |
            |--8.26%--tensor_get4d (inlined)
            |          conv_layer_backward (inlined)
            |          layer_backward (inlined)
            |          nn_backward (inlined)
            |          train_nn_batch_opt
            |          mnist_conv (inlined)
            |          main
            |          __libc_start_call_main
            |          __libc_start_main_impl (inlined)
            |          _start
            |
            |--4.77%--tensor_get3d (inlined)
            |          conv_layer_backward (inlined)
            |          layer_backward (inlined)
            |          nn_backward (inlined)
            |          train_nn_batch_opt
            |          mnist_conv (inlined)
            |          main
            |          __libc_start_call_main
            |          __libc_start_main_impl (inlined)
            |          _start
            |
            |--1.80%--tensor_add (inlined)
            |          linear_layer_backward (inlined)
            |          layer_backward (inlined)
            |          nn_backward (inlined)
            |          train_nn_batch_opt
            |          mnist_conv (inlined)
            |          main
            |          __libc_start_call_main
            |          __libc_start_main_impl (inlined)
            |          _start
            |
            |--1.66%--tensor_outer_product (inlined)
            |          linear_layer_backward (inlined)
            |          layer_backward (inlined)
            |          nn_backward (inlined)
            |          train_nn_batch_opt
            |          mnist_conv (inlined)
            |          main
            |          __libc_start_call_main
            |          __libc_start_main_impl (inlined)
            |          _start
            |
             --1.25%--step_adam (inlined)
                       optimizer_step (inlined)
                       train_nn_batch_opt
                       mnist_conv (inlined)
                       main
                       __libc_start_call_main
                       __libc_start_main_impl (inlined)
                       _start

     1.10%           119  neural_net_perf  libc.so.6          [.] __memset_avx512_unaligned_erms
            |
             --1.09%--__memset_avx512_unaligned_erms
                       |
                        --0.97%--train_nn_batch_opt
                                  mnist_conv (inlined)
                                  main
                                  __libc_start_call_main
                                  __libc_start_main_impl (inlined)
                                  _start

     0.38%            41  neural_net_perf  [kernel.kallsyms]  [k] __irqentry_text_end
     0.18%            19  neural_net_perf  neural_net_perf    [.] load_mnist_file
     0.17%            18  neural_net_perf  libc.so.6          [.] __memmove_avx512_unaligned_erms
     0.11%            14  neural_net_perf  [kernel.kallsyms]  [k] lock_vma_under_rcu
     0.11%            12  neural_net_perf  [kernel.kallsyms]  [k] srso_alias_return_thunk
     0.11%            12  neural_net_perf  [kernel.kallsyms]  [k] clear_page_erms
     0.10%            11  neural_net_perf  [kernel.kallsyms]  [k] sched_use_asym_prio
```

We can observe a large overhead with the indexing functions within the `conv_layer_backward` function.
The current versions of `conv_layer_forward` & `conv_layer_backward` are doing billions of index recalculations like so:

```c
for (int m = 0; m < layer->kernel_size; m++) {     // Kernel row
    for (int n = 0; n < layer->kernel_size; n++) { // Kernel col
        int h_idx = i * layer->stride + m;
        int w_idx = j * layer->stride + n;
        sum += tensor_get3d(X_pad, c, h_idx, w_idx) *
               tensor_get4d(layer->weights, o, c, m, n);
    }
}
```

Annotating and looking at our the hottest instruction:

```
        │      sum += tensor_get4d(layer->weights, o, c, m, n) *
   1.10 │        mov           0x10(%r13),%rsi  # Loading of layer->weights->strides
   2.25 │        mov           0x0(%r13),%rax # Loading of layer->weights->data
   1.08 │        mov           (%rsi),%r8d # Load strides[0] into r8d
   1.22 │        mov           0x4(%rsi),%esi # Load strides[1] into r8d
```

Although `tensor_get4d/3d` were inlined, the compiler cannot eliminate the dereferences of the `strides` and `data` fields,
even though they are constant across the iterations.
Examining the stride-based indexing approach for a 3D tensor, the offset for element (c, h, w) is:

    offset(c, h, w) = c * S_c + h * S_h + w * S_w

- `S_c` = H * W floats to move one channel
- `S_h` = W floats to move one row
- `S_w` = 1 float for column stride

These strides are invariant within the convolution computation, only depending on the tensor shape, which doesn't change in the forward or backward pass.
To optimize, we can hoist these invariant strides once before all the loops.
The optimized verisons can be found in [conv_layer.c](./src/nn/conv_layer.c), suffixed with `_stride_optimized`.
