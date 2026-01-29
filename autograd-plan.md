# Python Autograd System for nn_c

## Goal
Implement a Python autograd system to train MNIST, matching the C example in `src/examples/mnist_examples.c`:
- Network: Linear(784→128) → ReLU → Linear(128→10)
- Loss: Softmax Cross-Entropy
- Optimizer: SGD (Adam optional)

---

## Conceptual Background: Layer-Based vs Operator-Based Autograd

### Your Current C Approach (Layer-Based)
Each layer has explicit `forward()` and `backward()` methods:

```
Forward:  input → [Linear] → [ReLU] → [Linear] → output
                     ↓          ↓         ↓
                   cache      cache     cache

Backward: Each layer.backward() computes gradients explicitly
          Linear: dW = upstream ⊗ input^T, db = upstream, dinput = W^T @ upstream
          ReLU:   dinput = upstream * (input > 0)
```

**Pros**: Fast, predictable memory, no graph overhead
**Cons**: Must write backward() for every layer type, fixed structure

### Autograd Approach (Operator-Based)
Every operation builds a computation graph dynamically:

```
Forward (builds graph):
  x → [matmul] → [add] → [relu] → [matmul] → [add] → loss
        ↓         ↓        ↓         ↓         ↓
     grad_fn   grad_fn  grad_fn   grad_fn   grad_fn

Backward (traverses graph):
  loss.backward() walks graph in reverse, calling each grad_fn
```

**Key insight**: Instead of layers knowing how to differentiate themselves,
each *operation* (add, matmul, relu) knows how to compute gradients for its inputs.
The system chains these automatically via the computation graph.

**Pros**: Any composition of ops works automatically, dynamic control flow
**Cons**: Graph overhead, more memory for storing intermediate values

---

## Implementation Plan

### Phase 1: C Bindings (module.cpp)

Add these bindings to `python/bindings/module.cpp`:

**Tensor methods:**
- `scale(scalar)` → scaled tensor
- `subtract(other)` → a - b
- `elementwise_mul(other)` → element-wise multiplication
- `clone()` → deep copy
- `fill(val)` → fill in-place
- `sum()` → scalar sum
- `argmax()` → index of max element

**Backward helpers (static methods):**
- `Tensor.relu_backward(grad_output, input)` → gradient through ReLU
- `Tensor.sigmoid_backward(grad_output, sigmoid_out)` → gradient through sigmoid

**Module-level loss functions:**
- `softmax_cross_entropy(logits, target)` → scalar loss
- `softmax_cross_entropy_backward(logits, target)` → gradient tensor

**Fix matmul shape bug (existing):**
```cpp
// Current (WRONG): Tensor *dest = tensor_create(t->ndim, t->shape);
// Fixed: compute output shape [m, n] from inputs [m, k] @ [k, n]
```

### Phase 2: Core Autograd (new directory: `python/nn_c/autograd/`)

**`tensor.py` - AutogradTensor class:**
```python
class AutogradTensor:
    data: CTensor           # Underlying C tensor
    requires_grad: bool     # Track gradients?
    grad: CTensor | None    # Accumulated gradient
    grad_fn: GradFunction   # Function that created this (for graph)
    _is_leaf: bool          # True if created by user (not operation)

    def backward(grad_output=None)  # Trigger reverse-mode autodiff
    def zero_grad()                 # Clear accumulated gradient
```

**`function.py` - GradFunction base:**
```python
class GradFunction(ABC):
    saved_tensors: list     # Cached for backward
    inputs: list            # Parent AutogradTensors (graph edges)

    def save_for_backward(*tensors)
    def backward(grad_output) -> tuple[CTensor, ...]  # Compute input grads
```

**`ops.py` - Operation implementations:**
- `MatMulBackward`: dA = dC @ B^T, dB = A^T @ dC
- `AddBackward`: dA = dC, dB = dC (gradient passes through)
- `ReLUBackward`: dX = dY * (X > 0)
- Wrapper functions: `matmul()`, `add()`, `relu()` that build graph

**`engine.py` - Backward traversal:**
- Topological sort of computation graph
- Reverse traversal calling each `grad_fn.backward()`
- Accumulate gradients on leaf tensors

### Phase 3: Neural Network Modules (`python/nn_c/nn/`)

**`module.py` - Base class:**
```python
class Module:
    def forward(x) -> AutogradTensor
    def parameters() -> Iterator[AutogradTensor]
    def zero_grad()
```

**`linear.py` - Linear layer:**
- Weight: AutogradTensor with Xavier init, requires_grad=True
- Bias: AutogradTensor, requires_grad=True
- forward: `matmul(x, W.T) + b` using autograd ops

**`activations.py`:**
- `ReLU` module wrapping `relu()` op

**`container.py`:**
- `Sequential` for chaining modules

**`loss.py`:**
- `CrossEntropyLoss` - combines softmax + cross-entropy
- Returns AutogradTensor with proper grad_fn

### Phase 4: Optimizers (`python/nn_c/optim/`)

**`sgd.py`:**
```python
class SGD:
    def __init__(params, lr)
    def step():  # param.data -= lr * param.grad
    def zero_grad()
```

**`adam.py` (optional):**
- Maintains m (momentum) and v (velocity) per parameter
- Bias-corrected updates

### Phase 5: Data Loading (`python/nn_c/data/`)

**`mnist.py`:**
- Load MNIST from `datasets/mnist/` (same files as C)
- Return CTensors: X (60000, 784), Y (60000, 10) one-hot
- Helper: `get_row(tensor, i)` to extract samples

### Phase 6: Training Script

**`examples/train_mnist.py`:**
```python
model = Sequential(Linear(784, 128), ReLU(), Linear(128, 10))
criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters(), lr=0.1)

for epoch in range(10):
    for batch in batches:
        optimizer.zero_grad()
        for x, y in batch:
            loss = criterion(model(x), y)
            loss.backward()  # Accumulates gradients
        # Average and step
        optimizer.step()
```

---

## Files to Create/Modify

| File | Action |
|------|--------|
| `python/bindings/module.cpp` | Add new C bindings, fix matmul shape |
| `python/nn_c/autograd/__init__.py` | New - exports |
| `python/nn_c/autograd/tensor.py` | New - AutogradTensor |
| `python/nn_c/autograd/function.py` | New - GradFunction base |
| `python/nn_c/autograd/ops.py` | New - matmul, add, relu with backward |
| `python/nn_c/autograd/engine.py` | New - backward graph traversal |
| `python/nn_c/nn/module.py` | New - Module base class |
| `python/nn_c/nn/linear.py` | Modify - use autograd |
| `python/nn_c/nn/activations.py` | New - ReLU module |
| `python/nn_c/nn/container.py` | New - Sequential |
| `python/nn_c/nn/loss.py` | New - CrossEntropyLoss |
| `python/nn_c/optim/__init__.py` | New - exports |
| `python/nn_c/optim/sgd.py` | New - SGD optimizer |
| `python/nn_c/data/mnist.py` | New - MNIST loader |
| `python/nn_c/_nn_core/tensor.pyi` | Update stubs |

---

---

## Implementation Guidance (Hints)

### Phase 1 Hints: C Bindings

**Matmul shape fix** - The key insight:
```cpp
// For A[m,k] @ B[k,n] → C[m,n]
int m = t->shape[0];
int n = other->shape[1];
int new_shape[2] = {m, n};
Tensor *dest = tensor_create(2, new_shape);
```

**relu_backward pattern** - Look at `tensor_relu_derivative()` in `activations.h`:
```cpp
// It computes the mask (1 where input > 0, else 0)
// You need: grad_output * mask
```

### Phase 2 Hints: Core Autograd

**Start minimal** - Get this working first:
```python
# Test case: y = a + b, then y.backward()
a = AutogradTensor(CTensor.from_list([1,2,3], [3]), requires_grad=True)
b = AutogradTensor(CTensor.from_list([4,5,6], [3]), requires_grad=True)
y = add(a, b)
y.backward(CTensor.from_list([1,1,1], [3]))
# a.grad and b.grad should both be [1,1,1]
```

**Graph building** - The wrapper function pattern:
```python
def add(a: AutogradTensor, b: AutogradTensor) -> AutogradTensor:
    result_data = a.data.add(b.data)  # C operation

    if a.requires_grad or b.requires_grad:
        grad_fn = AddBackward()
        grad_fn.inputs = [a, b]  # Connect graph edges
        return AutogradTensor(result_data, requires_grad=True, grad_fn=grad_fn)

    return AutogradTensor(result_data, requires_grad=False)
```

**Topological sort** - Use DFS post-order:
```python
def topo_sort(root):
    visited, order = set(), []
    def dfs(node):
        if id(node) in visited: return
        visited.add(id(node))
        if node.grad_fn:
            for parent in node.grad_fn.inputs:
                dfs(parent)
        order.append(node)
    dfs(root)
    return order  # Forward order; reverse for backward
```

### Phase 3 Hints: Matmul Backward

This is the trickiest gradient. For C = A @ B:
- dL/dA = dL/dC @ B^T
- dL/dB = A^T @ dL/dC

**Watch dimensions carefully:**
```
A: [m, k]    B: [k, n]    C: [m, n]    dC: [m, n]
dA = dC @ B^T = [m,n] @ [n,k] = [m,k] ✓
dB = A^T @ dC = [k,m] @ [m,n] = [k,n] ✓
```

### Phase 4 Hints: Loss Function

**Softmax cross-entropy trick** - The derivative is beautifully simple:
```
dL/d(logits) = softmax(logits) - target_onehot
```
Your C code in `loss.c` already does this in `tensor_softmax_cross_entropy_derivative()`.

### Phase 5 Hints: SGD Optimizer

**The update rule:**
```python
for param in self.params:
    if param.grad is not None:
        # param = param - lr * grad
        scaled = param.grad.scale(self.lr)
        param.data = param.data.subtract(scaled)
```

### Phase 6 Hints: Debugging Gradients

**Numerical gradient check** - Gold standard for verifying backward:
```python
def numerical_grad(f, x, eps=1e-5):
    grad = []
    for i in range(x.size):
        x_plus = x.clone(); x_plus[i] += eps
        x_minus = x.clone(); x_minus[i] -= eps
        grad.append((f(x_plus) - f(x_minus)) / (2 * eps))
    return grad
```
Compare this to your autograd gradient - they should match to ~1e-4.

---

## Suggested Order of Implementation

1. **First**: Fix matmul shape bug in module.cpp (you can test immediately)
2. **Second**: Add `scale()`, `subtract()`, `clone()` bindings (needed for everything)
3. **Third**: `AutogradTensor` + `AddBackward` + engine (simplest case)
4. **Fourth**: `MatMulBackward` (this is the hard one - test thoroughly!)
5. **Fifth**: `ReLUBackward` + loss function bindings
6. **Sixth**: Module/Linear/SGD
7. **Finally**: MNIST training loop

---

## Verification

1. **Unit tests for autograd ops:**
   - `test_matmul_backward()` - numerical gradient check
   - `test_relu_backward()` - verify gradient mask
   - `test_graph_construction()` - verify topology

2. **Integration test:**
   - Train 1 epoch on 1000 samples
   - Verify loss decreases

3. **Full MNIST training:**
   - Run 10 epochs, batch_size=64, lr=0.1
   - Target: ~95% accuracy (matching C implementation)

4. **Compare with C:**
   ```bash
   # C version
   ./build/mnist_sgd

   # Python version
   uv run python examples/train_mnist.py
   ```
   Both should achieve similar accuracy.
