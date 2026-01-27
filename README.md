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

## Contributing

- Format code: `cmake --build build --target format`
- Run all tests before committing
- Update type stubs when adding bindings
