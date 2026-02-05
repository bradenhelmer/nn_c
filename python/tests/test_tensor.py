"""Tests for the Tensor class."""

import struct

from nn_c import Tensor


def test_tensor_creation():
    """Create tensor with shape, initialized to zeros."""
    t = Tensor([2, 3, 4])
    assert t.shape == (2, 3, 4)
    assert t.size == 24
    assert t.ndim == 3


def test_tensor_from_list():
    """Create tensor from flat Python list."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    t = Tensor.from_list(data, [2, 3])

    assert t.shape == (2, 3)
    assert t[0, 0] == 1.0
    assert t[1, 2] == 6.0


def test_tensor_from_bytes():
    """Create tensor from raw bytes (fastest path for data loading)."""
    # Pack 4 floats as bytes
    data = struct.pack("4f", 1.0, 2.0, 3.0, 4.0)
    t = Tensor.from_bytes(data, [2, 2])

    assert t.to_list() == [1.0, 2.0, 3.0, 4.0]


def test_tensor_roundtrip():
    """Verify data survives Python -> C -> Python."""
    original = [i * 0.1 for i in range(100)]
    t = Tensor.from_list(original, [10, 10])
    recovered = t.to_list()

    for a, b in zip(original, recovered):
        assert abs(a - b) < 1e-6


def test_tensor_indexing():
    """Element access and assignment."""
    t = Tensor([3, 3])

    # Flat indexing
    t[0] = 1.0
    t[4] = 5.0
    assert t[0] == 1.0
    assert t[4] == 5.0

    # Multi-dimensional indexing
    t[2, 2] = 9.0
    assert t[2, 2] == 9.0


def test_ones_like():
    """Test ones_like static method."""
    t = Tensor([2, 3])
    ones = Tensor.ones_like(t)
    assert ones.shape == (2, 3)
    assert all(ones[i] == 1.0 for i in range(ones.size))


def test_zeros_like():
    """Test zeros_like static method."""
    t = Tensor([2, 3])
    zeros = Tensor.zeros_like(t)
    assert zeros.shape == (2, 3)
    assert all(zeros[i] == 0.0 for i in range(zeros.size))


def test_tensor_repr():
    """Test tensor string representation."""
    t = Tensor([2, 3])
    assert repr(t) == "Tensor(shape=[2, 3])"

    t_grad = Tensor([2, 3], requires_grad=True)
    assert repr(t_grad) == "Tensor(shape=[2, 3], requires_grad=True)"


def test_tensor_add():
    """Test element-wise addition."""
    a = Tensor.from_list([1.0, 2.0, 3.0, 4.0], [2, 2])
    b = Tensor.from_list([5.0, 6.0, 7.0, 8.0], [2, 2])
    c = a.add(b)

    assert c.shape == (2, 2)
    assert c.to_list() == [6.0, 8.0, 10.0, 12.0]


def test_tensor_matmul():
    """Test matrix multiplication."""
    # 2x3 @ 3x2 = 2x2
    a = Tensor.from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    b = Tensor.from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [3, 2])
    c = a.matmul(b)

    assert c.shape == (2, 2)
    # Row 0: [1,2,3] @ [[1,2],[3,4],[5,6]] = [1*1+2*3+3*5, 1*2+2*4+3*6] = [22, 28]
    # Row 1: [4,5,6] @ [[1,2],[3,4],[5,6]] = [4*1+5*3+6*5, 4*2+5*4+6*6] = [49, 64]
    assert c[0, 0] == 22.0
    assert c[0, 1] == 28.0
    assert c[1, 0] == 49.0
    assert c[1, 1] == 64.0


def test_tensor_relu():
    """Test ReLU activation."""
    t = Tensor.from_list([-1.0, 0.0, 1.0, 2.0], [4])
    r = t.relu()

    assert r.to_list() == [0.0, 0.0, 1.0, 2.0]


def test_tensor_transpose2d():
    """Test 2D transpose."""
    t = Tensor.from_list([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3])
    tt = t.transpose2d()

    assert tt.shape == (3, 2)
    # Original: [[1,2,3],[4,5,6]]
    # Transposed: [[1,4],[2,5],[3,6]]
    assert tt[0, 0] == 1.0
    assert tt[0, 1] == 4.0
    assert tt[1, 0] == 2.0
    assert tt[2, 1] == 6.0
