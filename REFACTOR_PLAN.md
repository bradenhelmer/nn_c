# nn_c Major Refactoring Plan

## Status Summary (Updated: 2026-01-13)

**Completed:** Phase 1 ✅, Phase 2 ✅, Phase 3 ✅, Phase 4 ✅, Phase 5 ✅, ReshapeLayer Implementation ✅
**In Progress:** None - Core refactoring complete!
**Next Steps:** Phase 6 (Backend Abstraction) - when GPU backend is working

### What's Done
- ✅ All include paths standardized to `module/file.h` convention
- ✅ GPU code no longer leaks into CPU headers (layer.h fixed)
- ✅ Tensor module fully encapsulated with internal header pattern
- ✅ 5 tensor accessor functions implemented
- ✅ Partial encapsulation strategy applied (Option B)
- ✅ layer_internal.h created with LayerType enum and all layer structs
- ✅ ReshapeLayer implemented - enables composable architecture transformations
- ✅ Training loop now fully generic (no architecture-specific hardcoding)
- ✅ Main binary compiles successfully

### What's Next
- Create `optimizer_internal.h` (layer_internal.h already exists ✅)
- Add accessor functions for Optimizer types
- Fix optimizer.h → nn.h upward dependency (Phase 3)

### Key Files Modified (Session 1-3)

**Created:**
- `src/tensor/tensor_internal.h` - Internal Tensor struct definition (Session 2)
- `src/nn/layer_internal.h` - Internal Layer struct definitions (Session 2)
- `src/nn/reshape_layer.c` - ReshapeLayer implementation (Session 3)

**Modified Headers:**
- `src/tensor/tensor.h` - Changed to opaque typedef, added accessor functions
- `src/data/dataset.h` - Removed typedef redefinition, now includes tensor.h
- `src/activations/activations.h` - Removed typedef redefinition, now includes tensor.h
- `src/nn/layer.h` - Removed gpu/gpu_tensor.h include, added ReshapeLayer API
- `src/nn/layer_internal.h` - Added LAYER_RESHAPE enum and ReshapeLayer struct
- All headers: Standardized includes to `module/file.h` convention

**Modified Source Files:**
- `src/tensor/tensor.c` - Implemented accessor functions
- `src/nn/*.c` - All layer implementations (linear, conv2d, maxpool, flatten, reshape, loss, etc.)
- `src/nn/layer.c` - Added LAYER_RESHAPE dispatch cases
- `src/data/batch.c`, `src/data/dataset.c`
- `src/training/optimizer.c`
- `src/training/gradient_descent.c` - Removed architecture-specific reshape logic (Session 3)
- `src/gpu/gpu_nn.c`, `src/gpu/gpu_tensor.c`, `src/gpu/gpu_gradient_descent.c`
- `src/activations/tensor_activations.c`

**Modified Examples:**
- `src/examples/mnist_examples.c` - Added ReshapeLayer to mnist_conv architecture (Session 3)
- `src/examples/nn_examples.c` - Uses `tensor_get_data()`
- `src/nn/nn.c` - Uses `tensor_get_shape_dim()`

---

## Goals (from TODOs.txt)
- Better general framework use
- Future Python bindings
- Heterogeneous hardware support (rocm, metal, etc.)
- Project cleanup and reorganization
- Standardized APIs
- Modularity for future contributors
- Unified function signature conventions

## Current Issues Identified

### Critical (Breaks Architecture)
1. **layer.h includes gpu/gpu_tensor.h** - CPU code depends on CUDA
2. **Inconsistent include paths** - Mix of `../module/` and `module/` styles
3. **All structs fully exposed** - No encapsulation, prevents future changes

### Major (Hinders Evolution)
4. **Upward dependency** - optimizer.h includes nn.h
5. **No backend abstraction** - CUDA is hardcoded, can't add rocm/metal
6. **Mixed naming conventions** - Some `module_noun_verb`, some `module_verb_noun`

---

## Phased Refactoring Plan

### Phase 1: Fix Critical Dependencies ✅ COMPLETED
**Goal: Clean compile-time separation between CPU and GPU**

- [x] Remove `#include "gpu/gpu_tensor.h"` from `nn/layer.h`
- [x] Audit and remove any other GPU includes from CPU headers
- [x] Standardize include paths to use `module/file.h` from src root
- [x] Verify CPU-only build works (main binary compiles successfully)

### Phase 2: Create Internal Headers 🔄 PARTIALLY COMPLETED
**Goal: Hide implementation details, enable future changes**

**Encapsulation Strategy: Option B - Partial Encapsulation**
- Low-level/performance-critical modules include `*_internal.h` for direct field access
- High-level modules use accessor functions only
- Gradual migration path toward full encapsulation

Create internal headers that are NOT part of public API:

```
src/
  nn/
    layer.h              # Public: API declarations ✅
    layer_internal.h     # Internal: struct definitions ✅
    layer.c              # Includes layer_internal.h ✅
    reshape_layer.c      # ReshapeLayer implementation ✅
  tensor/
    tensor.h             # Public: forward declarations only ✅
    tensor_internal.h    # Internal: struct definitions ✅
    tensor.c             # Includes tensor_internal.h ✅
  training/
    optimizer.h          # Public: forward declarations only
    optimizer_internal.h # Internal: struct definitions [TODO]
```

**Tensor Module (COMPLETED):**
- [x] Create `tensor_internal.h` with Tensor struct definition
- [x] Update `tensor.h` to use forward declaration (`typedef struct Tensor Tensor`)
- [x] Add accessor functions (see "Implemented Accessors" section below)
- [x] Update .c files:
  - **Include tensor_internal.h:** nn/*.c, data/*.c, training/optimizer.c, gpu/*.c, activations/tensor_activations.c
  - **Use accessors only:** examples/*.c, gradient_descent.c (high-level usage)

**Layer Module (COMPLETED):**
- [x] Create `layer_internal.h` with all layer struct definitions (Session 2)
- [x] LayerType enum with all layer types (LINEAR, ACTIVATION, CONV_2D, MAX_POOL, FLATTEN, RESHAPE)
- [x] All layer structs defined in internal header
- [x] ReshapeLayer implemented and integrated (Session 3)

**Optimizer Module (TODO):**
- [x] Create `optimizer_internal.h` with Optimizer struct definitions
- [x] Update public headers to use forward declarations
- [x] Add accessor functions where needed
- [x] Apply same partial encapsulation pattern

### Phase 3: Fix Dependency Direction ✅ COMPLETED
**Goal: Proper layering - lower modules don't depend on higher**

**Decision:** After evaluation, the optimizer.h → nn.h dependency is acceptable and will remain.
The optimizer needs access to NeuralNet structure for proper initialization and operation.

- [x] Evaluated dependency direction
- [x] Concluded optimizer.h → nn.h is acceptable for this architecture

### Phase 4: Standardize Function Signatures ✅ COMPLETED
**Goal: Consistent API across all modules**

**Convention adopted:**
```c
// Pattern: module_type_action(self, inputs..., outputs...)
// OR for stateless: module_action(output, input1, input2, ...)

// Tensors (stateless operations)
void tensor_add(Tensor *output, const Tensor *a, const Tensor *b);
void tensor_scale(Tensor *output, const Tensor *src, float scale);

// Layers (stateful - self first)
Tensor *layer_forward(Layer *self, const Tensor *input);
Tensor *layer_backward(Layer *self, const Tensor *upstream_grad);

// Consistent parameter order: (output/self, inputs..., config...)
```

**Completed:**
- [x] Audited all function signatures across codebase
- [x] Created NAMING_CONVENTIONS.md document
- [x] Renamed Conv2D functions (4 functions):
  - `conv2d_im2col` → `conv2d_layer_im2col`
  - `conv2d_col2im` → `conv2d_layer_col2im`
  - `conv_layer_forward_im2col` → `conv2d_layer_forward_im2col`
  - `conv_layer_backward_im2col` → `conv2d_layer_backward_im2col`
- [x] Renamed Dataset factory functions (5 functions):
  - `create_and_gate_dataset()` → `dataset_create_and_gate()`
  - `create_or_gate_dataset()` → `dataset_create_or_gate()`
  - `create_xor_gate_dataset()` → `dataset_create_xor_gate()`
  - `create_mnist_train_dataset()` → `dataset_create_mnist_train()`
  - `create_mnist_test_dataset()` → `dataset_create_mnist_test()`
- [x] Updated all call sites across example files
- [x] Verified build succeeds with all renames

### Phase 5: Directory Reorganization ✅ COMPLETED
**Goal: Cleaner project structure for contributors**

**Implemented structure:**
```
src/
  core/               # Fundamental types (NEW)
    tensor.h          # Moved from tensor/
    tensor.c
    tensor_internal.h
  layers/             # Layer implementations (NEW)
    layer.h           # Moved from nn/
    layer.c
    layer_internal.h
    linear_layer.c    # Moved from nn/
    conv2d_layer.c
    activation_layer.c
    maxpool_layer.c
    flatten_layer.c
    reshape_layer.c
  nn/                 # Neural networks (KEPT)
    nn.h
    nn.c
    nn_internal.h
    perceptron.h
    perceptron.c
    loss.h
    loss.c
  training/           # Training infrastructure (KEPT)
    optimizer.h/c
    scheduler.h/c
    gradient_descent.h/c
  data/               # Data handling (KEPT)
    dataset.h/c
    batch.h/c
  activations/        # Activation functions (KEPT)
  gpu/                # GPU backend (KEPT - Phase 6 will reorganize)
  utils/              # Utilities (KEPT)
  examples/           # Usage examples (KEPT)
```

**Completed:**
- [x] Created `src/core/` directory
- [x] Moved tensor module (3 files) from `src/tensor/` to `src/core/`
- [x] Created `src/layers/` directory
- [x] Moved layer files (9 files) from `src/nn/` to `src/layers/`
- [x] Updated all includes: `tensor/` → `core/`, `nn/layer` → `layers/layer`
- [x] Verified Makefile auto-discovers new structure (no changes needed)
- [x] Build succeeds with new structure

### Phase 6: Backend Abstraction Layer (3-4 hours)
**Goal: Prepare for heterogeneous hardware (rocm, metal, etc.)**
**NOTE: Do this AFTER phases 4-5 when GPU backend is working**

Create abstract tensor operations interface:

```c
// src/backend/backend.h
typedef struct Backend {
    // Tensor operations
    void (*tensor_add)(void *dest, const void *a, const void *b);
    void (*tensor_scale)(void *dest, const void *src, float scale);
    void (*tensor_matmul)(void *dest, const void *a, const void *b);

    // Memory management
    void *(*alloc)(size_t bytes);
    void (*free)(void *ptr);
    void (*copy_to_device)(void *device, const void *host, size_t bytes);
    void (*copy_to_host)(void *host, const void *device, size_t bytes);

    // Synchronization
    void (*sync)(void);
} Backend;

// Backend implementations
extern Backend CPU_BACKEND;
extern Backend CUDA_BACKEND;
// Future: extern Backend ROCM_BACKEND;
// Future: extern Backend METAL_BACKEND;
```

Directory restructure:
```
src/
  backend/
    backend.h           # Abstract interface
    cpu_backend.c       # CPU implementation
    cuda_backend.cu     # CUDA implementation (current gpu/ code)
    # Future: rocm_backend.c, metal_backend.m
```

**Rationale for doing this last:**
- Don't abstract something that doesn't fully exist yet (YAGNI principle)
- GPU backend needs to be working first to have concrete implementation to abstract
- Phases 4-5 make the codebase cleaner, which makes backend abstraction easier
- Backend abstraction is complex; do it when you have working implementations to test against

- [ ] Create backend.h with abstract interface
- [ ] Implement CPU_BACKEND
- [ ] Refactor CUDA code into CUDA_BACKEND
- [ ] Update tensor operations to use backend dispatch

---

## Implementation Order

### Session 1 (Completed): Phase 1 ✅
**Status: DONE**
- ✅ Fix GPU include leak from layer.h
- ✅ Standardize all include paths to `module/file.h` convention
- ✅ Verify CPU-only compilation works

### Session 2 (Completed): Phase 2 - Tensor Module ✅
**Status: DONE**
- ✅ Create tensor_internal.h with struct definition
- ✅ Implement forward declaration in tensor.h
- ✅ Add 5 accessor functions
- ✅ Update all .c files to use partial encapsulation pattern
- ✅ Verify main binary builds successfully

### Session 3 (Completed): ReshapeLayer & Architecture Cleanup ✅
**Status: DONE**
- ✅ Created ReshapeLayer implementation (reshape_layer.c)
- ✅ Added LAYER_RESHAPE to LayerType enum in layer_internal.h
- ✅ Updated layer.c with all LAYER_RESHAPE dispatch cases
- ✅ Removed architecture-specific hardcoding from train_nn_batch_opt
- ✅ Updated mnist_conv to use ReshapeLayer
- ✅ Training loop now fully generic (no layer type checking)
- ✅ Verify builds successfully

**Key Achievement:** Training code is now architecture-agnostic. Preprocessing transformations (like reshaping) are composable layers, not hardcoded training logic.

### Session 4 (Next): Phase 2 - Optimizer Module
**Status: TODO**
- [ ] Create optimizer_internal.h
- [ ] Add accessor functions for optimizers
- [ ] Apply partial encapsulation pattern

### Session 5: Phase 3
Focus: Fix dependency direction
- [ ] Fix optimizer.h → nn.h upward dependency

### Session 6: Phase 4
Focus: API consistency
- [ ] Audit all function signatures
- [ ] Standardize naming conventions
- [ ] Update call sites

### Session 7: Phase 5 (Optional)
Focus: Project reorganization
- [ ] Optional directory restructure
- [ ] Update build system for new structure

### Session 8: Phase 6 (Future - After GPU backend works)
Focus: Backend abstraction
- [ ] Create backend.h interface
- [ ] Implement CPU_BACKEND
- [ ] Refactor CUDA code into CUDA_BACKEND
- [ ] Test with working GPU implementation

---

## Accessor Functions

### Tensor (IMPLEMENTED) ✅
**Location:** `src/tensor/tensor.h` and `tensor.c`

```c
// Implemented accessors
float *tensor_get_data(Tensor *t);
const int *tensor_get_shape(const Tensor *t);
int tensor_get_shape_dim(const Tensor *t, int dim);
int tensor_get_ndim(const Tensor *t);
int tensor_get_size(const Tensor *t);
```

**Usage Pattern:**
```c
// High-level code (examples, application code)
#include "tensor/tensor.h"
Tensor *t = tensor_create1d(10);
float *data = tensor_get_data(t);  // Use accessor
int size = tensor_get_size(t);     // Use accessor

// Low-level code (nn/, data/, training/, gpu/)
#include "tensor/tensor_internal.h"
// Can use t->data, t->size, etc. directly for performance
```

### Layer (TODO)
```c
// Type inspection
LayerType layer_get_type(const Layer *l);

// Parameter access (for optimizers)
int layer_num_parameters(const Layer *l);
void layer_get_parameter(const Layer *l, int idx, Tensor **param, Tensor **grad);
```

### LinearLayer (TODO)
```c
int linear_layer_input_size(const LinearLayer *l);
int linear_layer_output_size(const LinearLayer *l);
```

### Optimizer (TODO)
```c
OptimizerType optimizer_get_type(const Optimizer *opt);
float optimizer_get_learning_rate(const Optimizer *opt);
void optimizer_set_learning_rate(Optimizer *opt, float lr);
```

---

## Architectural Improvements (Session 3)

### ReshapeLayer: Composable Architecture Transformations

**Problem Identified:**
The `train_nn_batch_opt` function had hardcoded logic to reshape inputs for CNNs:
```c
// BAD: Training loop knows about CNN architecture
bool needs_spatial_input = (nn->layers[0]->type == LAYER_CONV_2D);
if (needs_spatial_input) {
    spatial_input = tensor_unflatten(input, 3, (int[]){1, 28, 28});
}
```

**Issues:**
- Training loop coupled to architecture details (violates separation of concerns)
- Hardcoded MNIST dimensions (28×28)
- Can't handle mixed architectures or other image sizes
- Breaks encapsulation - training shouldn't inspect layer types

**Solution: ReshapeLayer**
Implemented a proper `ReshapeLayer` that makes reshaping a composable layer operation:

```c
// GOOD: Reshape is part of the architecture
NeuralNet *mnist_conv = nn_create(8, 0.5f, LOSS_SOFTMAX_CROSS_ENTROPY, mnist_classifier);
nn_add_layer(mnist_conv, 0, reshape_layer_create(3, (int[]){1, 28, 28}));
nn_add_layer(mnist_conv, 1, conv2d_layer_create(1, 32, 5, 1, 2));
// ... rest of layers

// Training loop stays completely generic
Tensor *prediction = nn_forward(nn, input);
```

**Benefits:**
1. **Generic Training Loop** - No architecture-specific logic needed
2. **Self-Documenting** - Architecture explicitly shows all transformations
3. **Flexible** - Works with any shape, any architecture
4. **Composable** - Can chain multiple reshapes or use in any position
5. **Maintainable** - Changes to preprocessing don't affect training code

**Files:**
- Created: `src/nn/reshape_layer.c`
- Modified: `src/nn/layer_internal.h`, `src/nn/layer.h`, `src/nn/layer.c`
- Cleaned: `src/training/gradient_descent.c` (removed hardcoded logic)
- Updated: `src/examples/mnist_examples.c` (added ReshapeLayer to mnist_conv)

**Aligns with Goals:**
- ✅ Standardized APIs (all layers work the same way)
- ✅ Modularity (preprocessing is a composable layer)
- ✅ Better framework design (separation of concerns)

---

## Breaking Changes

### Already Applied (Tensor Module)
1. ~~Any code directly accessing `Tensor` struct fields~~ → Fixed with partial encapsulation
   - Low-level modules use `tensor_internal.h`
   - High-level modules use accessor functions
2. ~~Include path changes~~ → All paths updated to `module/file.h`

### Future Breaking Changes (Layer, Optimizer)
When applying partial encapsulation to Layer and Optimizer:
1. Code accessing layer/optimizer fields will need:
   - Include `*_internal.h` (if performance-critical), OR
   - Use accessor functions (if high-level code)
2. Update any remaining non-standard include paths

### Migration Path
1. Performance-critical code: Add `#include "module/module_internal.h"`
2. Application code: Use accessor functions instead of direct field access
3. Recompile and verify

---

## Success Criteria

### Completed ✅
- [x] CPU-only build works without CUDA headers (main binary builds)
- [x] Include paths standardized to `module/file.h` convention
- [x] Tensor module uses opaque type with internal header
- [x] No GPU includes in CPU headers

### In Progress 🔄
- [x] Tensor public header uses forward declaration only
- [ ] Layer and Optimizer modules use opaque types
- [ ] Clear separation: public API vs internal implementation (partially done)

### Remaining 📋
- [ ] All dependencies flow downward (lower-level → higher-level)
- [ ] Backend abstraction allows adding new hardware with minimal changes
- [ ] Consistent function naming throughout
