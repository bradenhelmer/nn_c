# AGENTS.md — nn_c

Neural network library written from scratch in C/CUDA with Python bindings (pybind11).

## Build System

CMake with scikit-build-core for Python packaging. Package manager is `uv`.

### Configure and Build

```bash
cmake -B build -DCMAKE_BUILD_TYPE=Release    # Configure (Release, Debug, Perf, Profile)
cmake --build build                           # Build the core C/CUDA library + Python extension
```

### Format

```bash
cmake --build build --target format    # clang-format on C/CUDA + ruff format on Python
```

### Tests

```bash
uv run pytest                           # Run all Python tests
uv run pytest python/tests/test_tensor.py::test_tensor_creation   # Run a single test
uv run pytest -k "matmul"              # Run tests matching a keyword
cmake --build build --target dev       # Rebuild bindings then run pytest
```

There is no C-level test runner. All tests are Python-side via pytest.

### Type Checking

```bash
uv run mypy         # Python type checking
```

### Python Environment

```bash
uv sync              # Install/sync all dependencies
uv pip install -e .  # Editable install of the Python package
```

## Project Layout

```
src/                     C/CUDA backend
  core/                  Tensor library (SIMD-optimized)
  layers/                Layer implementations (linear, conv2d, maxpool, activation, flatten, reshape)
  activations/           Scalar and tensor activation functions
  nn/                    NeuralNet composition, loss functions, perceptron
  training/              Optimizers (SGD, Momentum, Adam), gradient descent loops, LR schedulers
  data/                  Dataset loading, mini-batch iteration
  gpu/                   CUDA kernels and GPU execution paths
  utils/                 Utility functions, timing
  examples/              C example programs
python/
  bindings/module.cpp    pybind11 bindings
  nn_c/                  Python package (tensor, autograd, nn, optim, dataset)
  tests/                 pytest test suite
  examples/              Python examples
```

## C Code Style

### Formatting (.clang-format)

- Based on LLVM style, 4-space indentation, no tabs
- 100-column line limit
- Pointer alignment: right (`int *ptr`, not `int* ptr`)
- Opening braces on same line (Attach style)
- Always use braces (`InsertBraces: true`), no single-line if/loop/function

### Naming Conventions

| Element            | Convention              | Example                            |
|--------------------|-------------------------|------------------------------------|
| Functions          | `snake_case` + module prefix | `tensor_create()`, `nn_forward()` |
| Types/Structs      | `PascalCase`            | `Tensor`, `NeuralNet`, `GPUTensor` |
| Enums              | `UPPER_SNAKE_CASE` + type prefix | `LAYER_LINEAR`, `ACTIVATION_RELU` |
| Macros/Constants   | `UPPER_SNAKE_CASE`      | `BLOCK_SIZE`, `EPSILON`            |
| Internal/static    | `_underscore_prefix`    | `_tensor_set_size_metadata()`      |
| Local abbreviations| Short 2-3 letter names  | `ll` (LinearLayer), `cl` (Conv2DLayer) |
| Loop variables     | `i`, `j`, `k` or descriptive: `row`, `col`, `out_c`, `kh`, `kw` |

### File Organization

Each module uses a 3-file pattern:
- `module.h` — public API declarations
- `module_internal.h` — internal struct definitions (ownership documented in comments)
- `module.c` — implementation

### Include Order (in .c files)

1. Own internal header (`"tensor_internal.h"`)
2. Other project headers (`"core/tensor.h"`, `"utils/utils.h"`)
3. System/standard library headers (`<assert.h>`, `<stdio.h>`, `<stdlib.h>`, `<math.h>`)
4. SIMD intrinsic headers if needed (`<immintrin.h>`)

All project includes use paths relative to `src/`: `"core/tensor.h"`, `"layers/layer.h"`.

### Include Guards

```c
#ifndef TENSOR_H
#define TENSOR_H
/* ... */
#endif /* ifndef TENSOR_H */
```

### Header Comments

```c
/*
 * filename.c
 *
 * Brief description.
 */
```

Section dividers in headers:
```c
// =============================================================================
// SECTION TITLE
// =============================================================================
```

### Error Handling

- `assert()` for preconditions (dimension checks, index bounds, non-null pointers)
- `fprintf(stderr, ...) + exit(1)` for I/O failures (file loading)
- No custom error codes or error types in C code
- pybind11 layer throws `std::invalid_argument` / `std::out_of_range` for Python

### Memory Management

- Manual `malloc`/`free` with clear ownership. `_create()` returns owned pointer; caller must call `_free()`.
- Views use `owner = 0` flag to prevent double-free.
- NULL checks before free in cleanup paths.

### Polymorphism Pattern

Tagged-union dispatch: `Layer` has `LayerType type` enum + `void *layer` pointer. Generic functions (`layer_forward`, `layer_free`) use `switch` on type. Same pattern for `Optimizer`.

### Suppress Unused Warnings

Use `UNUSED` macro (`__attribute__((unused))`) on functions/params kept as reference implementations.

## Python Code Style

### Formatting

- `ruff format` with 100-char line limit (configured in `pyproject.toml`)
- Type hints on all function signatures
- `from __future__ import annotations` for forward references

### Module Docstrings

```python
"""
nn_c.module_name
~~~~~~~~~~~~~~~~
Brief description.
"""
```

### Conventions

- `@override` decorator on subclass methods
- `@final` on concrete classes
- `Protocol` for interfaces (e.g., `Optimizer`, `Dataset`)
- `__slots__` on performance-critical classes (e.g., `Tensor`)
- `__all__` exports in all `__init__.py` files
- NumPy-style docstrings with `Parameters` and `Returns` sections

### Test Style

```python
def test_tensor_creation():
    """Create tensor with shape, initialized to zeros."""
    t = Tensor([2, 3, 4])
    assert t.shape == (2, 3, 4)
```

- Function names: `test_<what_is_being_tested>`
- Docstring on each test function
- Plain `assert` statements

## Key Architecture Notes

- C99 for all C code, C++17 only for pybind11 bindings, CUDA with C++17
- CUDA targets architecture 89 (Ada Lovelace / RTX 40 series) — adjust `CMAKE_CUDA_ARCHITECTURES` for other GPUs
- Python `Tensor` wraps C `_CTensor` via pybind11 and layers autograd on top (pure Python)
- Two parallel training paths: C-native training loops and Python-side `Trainer` + autograd
- GPU path uses workspace-based memory allocation to minimize CUDA malloc calls
- AVX-512 SIMD used for CPU tensor operations
- Headers needing C++ access include `extern "C" { ... }` guards
