# nn_c Naming Conventions

## Status: Phase 4 - Standardization in Progress

This document defines the naming conventions for the nn_c neural network library to ensure consistency and maintainability.

---

## Core Principles

1. **Snake Case:** All function and variable names use `snake_case`
2. **Module Prefixes:** All public functions use their module name as a prefix
3. **Const Correctness:** Input parameters that aren't modified are marked `const`
4. **Destination First:** Output parameters come before input parameters
5. **Self First:** Object methods receive the object as the first parameter

---

## Function Naming Patterns

### Pattern 1: Stateless Operations (Tensor, Activations, Loss)

**Format:** `module_operation(output, input1, input2, ...)`

Output/destination parameter comes FIRST, followed by inputs, then scalars.

```c
// Tensor operations
void tensor_add(Tensor *dest, const Tensor *a, const Tensor *b);
void tensor_scale(Tensor *dest, const Tensor *src, float scalar);
void tensor_matmul(Tensor *dest, const Tensor *a, const Tensor *b);

// Activation operations
void tensor_sigmoid(Tensor *output, const Tensor *input);
void tensor_relu(Tensor *output, const Tensor *input);
float sigmoid_scalar(float x);
float relu_scalar(float x);

// Loss operations
void tensor_mse_derivative(Tensor *result, const Tensor *prediction, const Tensor *target);
float tensor_mse(const Tensor *prediction, const Tensor *target);
```

**Rationale:** Makes it clear what gets modified and enables chaining operations naturally.

---

### Pattern 2: Stateful Objects (Layers, Networks, Optimizers)

**Format:** `module_action(self, inputs...)`

The object being operated on comes FIRST, followed by inputs.

```c
// Generic layer operations
Tensor *layer_forward(Layer *layer, const Tensor *input);
Tensor *layer_backward(Layer *layer, const Tensor *upstream_grad);

// Neural network operations
Tensor *nn_forward(NeuralNet *nn, const Tensor *input);
void nn_backward(NeuralNet *nn, const Tensor *target);

// Optimizer operations
void optimizer_step(Optimizer *opt, NeuralNet *nn);
void optimizer_set_lr(Optimizer *opt, float lr);
```

**Rationale:** Object-oriented style makes it clear which object the method operates on.

---

### Pattern 3: Type-Specific Operations

**Format:** `type_module_action(self, inputs...)`

For specific layer types or specialized operations.

```c
// Linear layer
LinearLayer *linear_layer_create(int input_size, int output_size);
Tensor *linear_layer_forward(LinearLayer *layer, const Tensor *input);
void linear_layer_init_xavier(LinearLayer *layer);

// Conv2D layer
Conv2DLayer *conv2d_layer_create(int in_channels, int out_channels, int kernel_size, int stride, int padding);
Tensor *conv2d_layer_forward(Conv2DLayer *layer, const Tensor *input);
void conv2d_layer_init_weights(Conv2DLayer *layer, float std);

// Activation layer
ActivationLayer *activation_layer_create(Activation activation);
Tensor *activation_layer_forward(ActivationLayer *layer, const Tensor *input);
```

**Rationale:** Distinguishes type-specific operations from generic operations on base types.

---

### Pattern 4: Factory Functions

**Format:** `module_create_variant(params...)` or `module_create(params...)`

```c
// Generic creation
Dataset *dataset_create(int num_samples, int num_features);
Tensor *tensor_create(int ndim, const int *shape);

// Typed creation
Optimizer *optimizer_create_sgd(float learning_rate);
Optimizer *optimizer_create_momentum(float learning_rate, float momentum);
Optimizer *optimizer_create_adam(float learning_rate, float beta1, float beta2, float epsilon);

// Predefined datasets - STANDARDIZED
Dataset *dataset_create_and_gate(void);
Dataset *dataset_create_or_gate(void);
Dataset *dataset_create_xor_gate(void);
Dataset *dataset_create_mnist_train(void);
Dataset *dataset_create_mnist_test(void);
```

**Rationale:** Consistent `module_create` prefix makes factory functions immediately recognizable.

---

### Pattern 5: Accessors and Mutators

**Format:** `module_get_property()` and `module_set_property()`

```c
// Getters (const correctness)
float *tensor_get_data(Tensor *t);
const int *tensor_get_shape(const Tensor *t);
int tensor_get_ndim(const Tensor *t);
int nn_get_num_layers(const NeuralNet *nn);
float optimizer_get_lr(const Optimizer *opt);

// Setters
void optimizer_set_lr(Optimizer *opt, float lr);
void tensor_set2d(Tensor *t, int i, int j, float value);
```

**Rationale:** Standard getter/setter pattern familiar to all C programmers.

---

## Current Inconsistencies and Fixes

### Issue 1: Conv2D Mixed Naming Patterns

**Current State:**
```c
// Three different patterns!
Conv2DLayer *conv2d_layer_create(...);           // ✅ CORRECT
void conv2d_layer_init_weights(...);             // ✅ CORRECT
void conv2d_im2col(...);                         // ❌ Missing "layer" infix
void conv_layer_forward_im2col(...);             // ❌ "conv" instead of "conv2d"
```

**Fix:** Standardize all to `conv2d_layer_*` pattern
```c
// After standardization
Conv2DLayer *conv2d_layer_create(...);           // No change
void conv2d_layer_init_weights(...);             // No change
void conv2d_layer_im2col(...);                   // RENAME: add "layer"
void conv2d_layer_forward_im2col(...);           // RENAME: conv → conv2d
void conv2d_layer_backward_im2col(...);          // RENAME: conv → conv2d
```

---

### Issue 2: Dataset Factory Function Reversal

**Current State:**
```c
Dataset *dataset_create(int num_samples, int num_features);  // ✅ CORRECT
Dataset *create_and_gate_dataset(void);                       // ❌ REVERSED
Dataset *create_or_gate_dataset(void);                        // ❌ REVERSED
Dataset *create_mnist_train_dataset(void);                    // ❌ REVERSED
```

**Fix:** Use consistent `dataset_create_*` pattern
```c
// After standardization
Dataset *dataset_create(int num_samples, int num_features);   // No change
Dataset *dataset_create_and_gate(void);                       // RENAME
Dataset *dataset_create_or_gate(void);                        // RENAME
Dataset *dataset_create_xor_gate(void);                       // RENAME
Dataset *dataset_create_mnist_train(void);                    // RENAME
Dataset *dataset_create_mnist_test(void);                     // RENAME
```

---

### Issue 3: Loss Function Naming (Minor)

**Current State:**
```c
float mse_loss(float predicted, float target);               // Scalar version
float tensor_mse(const Tensor *prediction, const Tensor *target);  // Tensor version
```

**Observation:** Inconsistent - scalar has `_loss` suffix, tensor doesn't.

**Options:**
1. **Option A (Recommended):** Keep as-is - clear enough in context
2. **Option B:** Add `_loss` to tensor versions: `tensor_mse_loss()`
3. **Option C:** Add `_scalar` to scalar versions: `mse_loss_scalar()`

**Decision:** Keep as-is for now. The distinction between scalar and tensor is clear from context.

---

## Standardization Rules Summary

### ✅ Currently Consistent (Keep These)

1. **Tensor Module:** All functions follow `tensor_operation` pattern
2. **Layer Module:** Generic functions use `layer_action`, specific use `type_layer_action`
3. **NN Module:** All functions follow `nn_action` pattern
4. **Optimizer Module:** All functions follow `optimizer_action` or `optimizer_create_type`
5. **Activations Module:** Clear separation between `*_scalar()` and `tensor_*()` functions
6. **Batch Module:** Clear separation between `batch_*` and `batch_iterator_*`

### 🔧 Needs Fixing

1. **Conv2D Functions:** Standardize to `conv2d_layer_*` pattern (5 functions)
2. **Dataset Factories:** Standardize to `dataset_create_*` pattern (6 functions)

---

## Parameter Ordering Rules

### Rule 1: Output Parameters First
When a function modifies an output parameter, it comes FIRST.

```c
// ✅ CORRECT
void tensor_add(Tensor *output, const Tensor *a, const Tensor *b);
void tensor_sigmoid(Tensor *output, const Tensor *input);

// ❌ INCORRECT
void tensor_add(const Tensor *a, const Tensor *b, Tensor *output);
```

### Rule 2: Self Parameter First (Object Methods)
For object methods, the object comes FIRST.

```c
// ✅ CORRECT
Tensor *layer_forward(Layer *layer, const Tensor *input);
void optimizer_step(Optimizer *opt, NeuralNet *nn);

// ❌ INCORRECT
Tensor *layer_forward(const Tensor *input, Layer *layer);
```

### Rule 3: Order of Multiple Inputs
When multiple inputs are needed:
1. Primary input (e.g., predictions)
2. Secondary input (e.g., targets)
3. Scalar parameters (e.g., learning_rate)
4. Configuration structs

```c
// ✅ CORRECT
void tensor_mse_derivative(Tensor *result, const Tensor *prediction, const Tensor *target);
Tensor *nn_forward(NeuralNet *nn, const Tensor *input);

// ✅ CORRECT - scalars last
void tensor_scale(Tensor *dest, const Tensor *src, float scalar);
```

---

## Const Correctness Rules

### Rule 1: Input Parameters
Input parameters that aren't modified MUST be marked `const`.

```c
// ✅ CORRECT
void tensor_add(Tensor *dest, const Tensor *a, const Tensor *b);
float tensor_mse(const Tensor *prediction, const Tensor *target);
int dataset_get_num_samples(const Dataset *d);

// ❌ INCORRECT
void tensor_add(Tensor *dest, Tensor *a, Tensor *b);
```

### Rule 2: Return Types
Functions that return pointers to internal data should return `const` if the data shouldn't be modified externally.

```c
// ✅ CORRECT
const int *tensor_get_shape(const Tensor *t);  // Don't modify shape array

// ✅ CORRECT
float *tensor_get_data(Tensor *t);  // Allow modification of data
```

---

## Migration Plan

### Phase 1: Rename Conv2D Functions ✅
Files to modify:
- `src/nn/conv2d_layer.c` (implementations)
- `src/nn/layer.h` (declarations)
- All call sites in layer.c, gpu files

**Renames:**
```c
conv2d_im2col         → conv2d_layer_im2col
conv2d_col2im         → conv2d_layer_col2im
conv_layer_forward_im2col  → conv2d_layer_forward_im2col
conv_layer_backward_im2col → conv2d_layer_backward_im2col
```

### Phase 2: Rename Dataset Factories ✅
Files to modify:
- `src/data/dataset.h` (declarations)
- `src/data/dataset.c` (implementations)
- All example files that use these functions

**Renames:**
```c
create_and_gate_dataset       → dataset_create_and_gate
create_or_gate_dataset        → dataset_create_or_gate
create_xor_gate_dataset       → dataset_create_xor_gate
create_mnist_train_dataset    → dataset_create_mnist_train
create_mnist_test_dataset     → dataset_create_mnist_test
```

---

## Future Considerations

### Potential Improvements (Not Implemented Yet)

1. **Error Handling Convention**
   - Consider returning error codes or using out-parameters for error reporting
   - Example: `int tensor_create_checked(Tensor **out, int ndim, const int *shape)`

2. **GPU Function Naming**
   - Current: `gpu_*` prefix for all GPU functions
   - Consider: Separate backend abstraction (Phase 6)

3. **Internal vs Public APIs**
   - Current: Use `*_internal.h` headers for internal structs
   - Working well - keep this pattern

---

## Enforcement

### For New Code
1. All new functions MUST follow these conventions
2. Code review should check for naming consistency
3. Use this document as reference during development

### For Existing Code
1. Prioritize fixing inconsistencies that affect public API
2. Update documentation to match actual function names
3. Consider compatibility if used by external code

---

## Examples by Module

### Tensor Module ✅ (Already Consistent)
```c
// Creation
Tensor *tensor_create(int ndim, const int *shape);
Tensor *tensor_create1d(int size);
Tensor *tensor_zeros(int ndim, const int *shape);

// Operations (output first)
void tensor_add(Tensor *dest, const Tensor *a, const Tensor *b);
void tensor_matmul(Tensor *dest, const Tensor *a, const Tensor *b);
void tensor_scale(Tensor *dest, const Tensor *src, float scalar);

// Accessors (const correctness)
float *tensor_get_data(Tensor *t);
const int *tensor_get_shape(const Tensor *t);
int tensor_get_size(const Tensor *t);

// Cleanup
void tensor_free(Tensor *t);
```

### Layer Module ✅ (Mostly Consistent, Conv2D needs fixes)
```c
// Generic operations
Tensor *layer_forward(Layer *layer, const Tensor *input);
Tensor *layer_backward(Layer *layer, const Tensor *upstream_grad);
void layer_free(Layer *layer);

// Type-specific: Linear Layer
LinearLayer *linear_layer_create(int input_size, int output_size);
Tensor *linear_layer_forward(LinearLayer *layer, const Tensor *input);
void linear_layer_init_xavier(LinearLayer *layer);
void linear_layer_free(LinearLayer *layer);

// Type-specific: Conv2D Layer (AFTER Phase 4 fixes)
Conv2DLayer *conv2d_layer_create(int in_channels, int out_channels, int kernel_size, int stride, int padding);
Tensor *conv2d_layer_forward(Conv2DLayer *layer, const Tensor *input);
void conv2d_layer_init_weights(Conv2DLayer *layer, float std);
void conv2d_layer_im2col(const Conv2DLayer *layer, const Tensor *input, Tensor *col);
Tensor *conv2d_layer_forward_im2col(Conv2DLayer *layer, const Tensor *input);
void conv2d_layer_free(Conv2DLayer *layer);
```

### Dataset Module 🔧 (Factories need renaming)
```c
// Generic creation
Dataset *dataset_create(int num_samples, int num_features);

// Specific datasets (AFTER Phase 4 fixes)
Dataset *dataset_create_and_gate(void);
Dataset *dataset_create_or_gate(void);
Dataset *dataset_create_xor_gate(void);
Dataset *dataset_create_mnist_train(void);
Dataset *dataset_create_mnist_test(void);

// Accessors
int dataset_get_num_samples(const Dataset *d);
Tensor *dataset_get_X(const Dataset *d);
Tensor *dataset_get_Y(const Dataset *d);

// Cleanup
void dataset_free(Dataset *d);
```

---

## Rationale

### Why Module Prefixes?
C lacks namespaces, so module prefixes prevent naming collisions and make code self-documenting.

### Why Destination First?
Common in C libraries (memcpy, strcpy, etc.) and makes the flow of data clear when reading left-to-right.

### Why Self First for Objects?
Mimics object-oriented method calls, making it clear which object is being operated on.

### Why Const Correctness?
Documents intent, enables compiler optimizations, and prevents accidental mutations.

---

## References

This convention is inspired by:
- Standard C library conventions (memcpy, strcpy, etc.)
- NumPy C API conventions
- GTK/GLib naming conventions
- Linux kernel style for C

---

**Last Updated:** 2026-01-13
**Status:** Phase 4 in progress - Conv2D and Dataset renames pending
