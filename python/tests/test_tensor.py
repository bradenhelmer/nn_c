"""Example tests showing how to use the bindings."""

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
