# nn_c Major Refactoring Plan

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

### Phase 1: Fix Critical Dependencies (1-2 hours)
**Goal: Clean compile-time separation between CPU and GPU**

- [ ] Remove `#include "gpu/gpu_tensor.h"` from `nn/layer.h`
- [ ] Audit and remove any other GPU includes from CPU headers
- [ ] Standardize include paths to use `module/file.h` from src root
- [ ] Verify CPU-only build works

### Phase 2: Create Internal Headers (2-3 hours)
**Goal: Hide implementation details, enable future changes**

Create internal headers that are NOT part of public API:

```
src/
  nn/
    layer.h              # Public: forward declarations only
    layer_internal.h     # Internal: struct definitions
    layer.c              # Includes layer_internal.h
  tensor/
    tensor.h             # Public: forward declarations only
    tensor_internal.h    # Internal: struct definitions
    tensor.c             # Includes tensor_internal.h
  training/
    optimizer.h          # Public: forward declarations only
    optimizer_internal.h # Internal: struct definitions
```

Changes needed:
- [ ] Create `tensor_internal.h` with Tensor struct definition
- [ ] Create `layer_internal.h` with all layer struct definitions
- [ ] Create `optimizer_internal.h` with Optimizer struct
- [ ] Update public headers to use forward declarations
- [ ] Add accessor functions where direct field access is needed
- [ ] Update all .c files to include internal headers

### Phase 3: Fix Dependency Direction (1 hour)
**Goal: Proper layering - lower modules don't depend on higher**

```
Current:  optimizer.h → nn.h  (upward - BAD)
Target:   nn.h uses optimizer through abstract interface
```

- [ ] Remove nn.h include from optimizer.h
- [ ] Use forward declaration: `struct NeuralNet;`
- [ ] optimizer_init() only needs pointer, not full definition

### Phase 4: Backend Abstraction Layer (3-4 hours)
**Goal: Prepare for heterogeneous hardware (rocm, metal, etc.)**

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

- [ ] Create backend.h with abstract interface
- [ ] Implement CPU_BACKEND
- [ ] Refactor CUDA code into CUDA_BACKEND
- [ ] Update tensor operations to use backend dispatch

### Phase 5: Standardize Function Signatures (1-2 hours)
**Goal: Consistent API across all modules**

Convention to adopt:
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

- [ ] Audit all function signatures
- [ ] Create naming convention document
- [ ] Rename inconsistent functions
- [ ] Update all call sites

### Phase 6: Directory Reorganization (Optional, 2-3 hours)
**Goal: Cleaner project structure for contributors**

Proposed structure:
```
src/
  core/               # Fundamental types
    tensor.h/c
    tensor_internal.h
  layers/             # Layer implementations
    layer.h           # Public API
    layer_internal.h
    linear.c
    conv2d.c
    activation.c
    maxpool.c
    flatten.c
  training/           # Training infrastructure
    optimizer.h/c
    scheduler.h/c
    trainer.h/c       # Renamed from gradient_descent
  data/               # Data handling
    dataset.h/c
    batch.h/c
  backend/            # Hardware backends
    backend.h
    cpu/
    cuda/
  utils/              # Utilities
    timing.h/c
    memory.h/c        # Future: arena allocator
  examples/           # Usage examples
```

---

## Implementation Order

### Session 1 (Today): Phases 1-2
Focus: Critical fixes and encapsulation foundation
- Fix GPU include leak
- Create internal headers
- Make key types opaque

### Session 2: Phases 3-4
Focus: Architectural cleanup
- Fix dependency direction
- Create backend abstraction

### Session 3: Phases 5-6
Focus: Polish and reorganization
- Standardize naming
- Optional directory restructure

---

## Accessor Functions Needed

When making types opaque, we need accessor functions:

### Tensor
```c
// Read-only accessors
int tensor_ndim(const Tensor *t);
int tensor_size(const Tensor *t);
const int *tensor_shape(const Tensor *t);
const float *tensor_data_const(const Tensor *t);

// Mutable accessor (for operations that need to modify)
float *tensor_data(Tensor *t);
```

### Layer
```c
// Type inspection
LayerType layer_get_type(const Layer *l);

// Parameter access (for optimizers)
int layer_num_parameters(const Layer *l);
void layer_get_parameter(const Layer *l, int idx, Tensor **param, Tensor **grad);
```

### LinearLayer (if exposed)
```c
int linear_layer_input_size(const LinearLayer *l);
int linear_layer_output_size(const LinearLayer *l);
```

---

## Breaking Changes

This refactor will break:
1. Any code directly accessing struct fields (e.g., `tensor->data[i]`)
2. Any code including internal headers
3. Code assuming specific include paths

Migration path:
1. Replace direct field access with accessor functions
2. Update includes to use public headers only
3. Recompile

---

## Success Criteria

After refactoring:
- [ ] CPU-only build works without CUDA headers
- [ ] No public header exposes struct internals
- [ ] All dependencies flow downward (lower-level → higher-level)
- [ ] Backend abstraction allows adding new hardware with minimal changes
- [ ] Consistent function naming throughout
- [ ] Clear separation: public API vs internal implementation
