# nn_c

Neural networks from scratch in C/CUDA with Python bindings.

High-performance neural network library with a C/CUDA backend, designed to be used primarily through Python.

## Features

- C/CUDA backend for high-performance tensor operations
- Clean Python API via pybind11
- GPU acceleration with CUDA
- Modern build system (CMake + scikit-build-core)

## Prerequisites

- C compiler (Clang/GCC)
- CUDA Toolkit 11+
- Python 3.12+
- CMake 3.18+
- [uv](https://docs.astral.sh/uv/) package manager

## Quick Start

```bash
git clone <repository-url>
cd nn_c
uv sync
uv pip install -e .
python -c "from nn_c import Tensor; print(Tensor([2, 3]))"
```

## Basic Usage

```python
from nn_c import Tensor

# Create tensors
t = Tensor([2, 3])

# From Python list
data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
t = Tensor.from_list(data, [2, 3])

# Access and inspect
print(t[0, 0])  # 1.0
print(t.shape)  # (2, 3)
print(t.size)   # 6

# Serialize
bytes_data = t.to_bytes()
t2 = Tensor.from_bytes([2, 3], bytes_data)
```

## Development

```bash
# Setup (one-time)
uv sync
uv pip install -e .

# After C/CUDA changes
cmake --build build

# Run tests
python python/run_tests.py
```

## Implementation Details

### im2col Convolution

The CPU convolution path uses an im2col transform to reduce the 6-nested-loop naive convolution
to a single GEMM call. The key function is `conv2d_layer_im2col` in
[conv2d_layer.c](src/layers/conv2d_layer.c).

**What it does mathematically:** For an input of shape `(C_in, H, W)` with a kernel of size
`K×K`, im2col extracts every `C_in × K × K` patch that the kernel slides over and lays them
out as columns in a matrix `X_col` of shape `(C_in*K*K) × (H_out*W_out)`. The weight tensor
is reshaped to `W_row` of shape `(C_out) × (C_in*K*K)`. The full convolution output is then:

    Y_flat = W_row × X_col       // (C_out) × (H_out*W_out)

which is a single GEMM — no spatial traversal logic, no strided index arithmetic in the
inner loop.

**Why this matters for cache locality:** The naive nested loop implementation accesses both
the input and kernel tensors with irregular strides — for each output position `(i, j)`, the
kernel slides over a non-contiguous `K×K` region of the input. im2col pays a one-time cost
to materialize `X_col` (a contiguous layout of all input patches), after which GEMM's inner
loop reads memory sequentially. On large kernels and input sizes, the sequential access
pattern drives cache utilization significantly higher. Measured speedup over the naive
implementation: **5-10x**.

**Backward pass:** The inverse operation, `col2im`, scatters gradient values from
`dX_col = W^T × dY_flat` back to the padded input gradient using accumulated additions into
the overlapping patch regions.

---

### AVX-512 GEMM (CPU Path)

Both the matrix-vector product and general matrix multiply paths in
[tensor.c](src/core/tensor.c) are manually vectorized using AVX-512 intrinsics.

- **Width:** 16 single-precision floats per `__m512` register.
- **FMA:** `_mm512_fmadd_ps` fuses multiply and accumulate into a single instruction,
  halving the number of floating-point operations relative to separate `mul` + `add`.
- **GEMM tiling:** The GEMM kernel tiles across the output columns to keep `__m512`
  accumulators register-resident across multiple inner iterations. The `_mm512_set1_ps`
  broadcast loads a scalar `a[row][inner]` into all 16 lanes, then a single `fmadd` with
  the 16-wide `b[inner][col..]` strip accumulates 16 output elements in one instruction.
- **Tail handling:** Columns that don't fill a full 16-wide vector fall through to a scalar
  loop, so correctness holds for arbitrary matrix dimensions.

---

### Warp-Level Reductions (GPU Path)

The softmax cross-entropy kernel in [gpu_loss.cu](src/gpu/gpu_loss.cu) uses warp shuffle
instructions to reduce across 32 threads without touching shared memory.

**The pattern (`warp_reduce_sum`, `warp_reduce_max`):**
```c
for (int offset = 16; offset > 0; offset >>= 1)
    val += __shfl_down_sync(0xFFFFFFFF, val, offset);
```
This is a butterfly tree reduction: each thread receives a value from the thread `offset`
lanes ahead using a register-to-register shuffle. After 5 iterations (offsets 16, 8, 4, 2,
1), lane 0 holds the reduction result across all 32 lanes. No shared memory is used within
a warp — all communication is register-level.

**Block-level reduction:** When a block spans multiple warps, each warp's lane-0 writes its
partial result to a 32-element `__shared__` array. Warp 0 then re-reduces those values with
a second warp-level shuffle. This limits shared memory usage to 32 floats per block
(128 bytes) rather than one float per thread.

**Why this matters:** The conventional block reduction allocates one shared memory slot per
thread (256 floats = 1 KB per block). The warp-first approach reduces that to 32 floats
(128 bytes), improving occupancy and eliminating the `__syncthreads` barrier on every round
of the reduction. The softmax forward kernel runs this reduction twice per sample (once for
max, once for sum-of-exp), so the savings compound across the batch.

---

### 256-Byte Aligned Workspace Allocator

All GPU intermediate activations and gradients during a forward/backward pass are allocated
from a pre-allocated device memory arena. The arena is set up once at network initialization
(`gpu_nn_create`) with a single `cudaMalloc` and reused every training step.

**Allocation (`workspace_alloc` in [gpu_nn.c](src/gpu/gpu_nn.c)):**
```c
size_t aligned = (bytes + 255) & ~((size_t)255);
float *ptr = (float *)((char *)gpu_nn->d_workspace + gpu_nn->workspace_offset);
gpu_nn->workspace_offset += aligned;
```

**Why 256 bytes:** CUDA global memory accesses are coalesced when each warp's 32 threads
access a 128-byte aligned contiguous region. Using 256-byte alignment (2× the hardware
minimum) ensures that any sub-allocation whose size is a multiple of 4 bytes begins on a
boundary that is valid for both the current warp and the next, preventing false sharing
across warp boundaries regardless of access pattern.

**Why a bump allocator:** `cudaMalloc` and `cudaFree` are expensive — each call
synchronizes the CUDA context. Calling them per-tensor per-step would serialize the GPU
with the CPU on every layer's activation. The workspace resets by zeroing a single
`size_t` offset (`workspace_reset`), making per-step allocation effectively free.

---

### Fused SGD Update Kernel

The SGD optimizer implements a fused gradient-scaling + weight-update kernel
(`_sgd_update_scaled_kernel` in [gpu_optimizer.cu](src/gpu/gpu_optimizer.cu)):

```c
// lr_scaled = learning_rate * grad_scale, computed once on CPU
weights[idx] -= lr_scaled * grads[idx];
```

The unfused path requires two passes: one kernel to scale all gradients in-place, then a
second kernel to apply the update. The fused version computes `lr * grad_scale` as a scalar
on the CPU (one multiply, free), passes it as a kernel argument, and each thread applies
both operations in a single global memory read-modify-write. This halves the number of
kernel launches and eliminates one full read pass over the gradient buffer.

---

### Performance

| Operation | Variant | Notes |
| --- | --- | --- |
| Conv2D forward | im2col + GEMM | 5-10x over naive 6-loop implementation |
| Conv2D forward | AVX-512 GEMM (CPU) | 16-wide FMA on inner GEMM loop |
| Softmax cross-entropy | Warp shuffle reduction | No per-thread shared memory; 128-byte shared per block |
| Training step allocation | Workspace bump allocator | Zero `cudaMalloc` calls after initialization |

A formal micro-benchmark suite is a planned addition — see [TODOs.txt](TODOs.txt).

---

## Contributing

- Format code: `cmake --build build --target format`
- Run all tests before committing
- Update type stubs when adding bindings
