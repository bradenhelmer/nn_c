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


# =============================================================================
# Example: Loading MNIST with from_bytes
# =============================================================================


def load_mnist_images(filepath: str) -> Tensor:
    """
    Load MNIST images from IDX file format.

    This demonstrates the intended data loading pattern:
    Python parses the format, then uses from_bytes for fast transfer to C.
    """
    with open(filepath, "rb") as f:
        # IDX file header
        magic = struct.unpack(">I", f.read(4))[0]
        assert magic == 2051, f"Invalid magic number: {magic}"

        num_images = struct.unpack(">I", f.read(4))[0]
        rows = struct.unpack(">I", f.read(4))[0]
        cols = struct.unpack(">I", f.read(4))[0]

        # Read raw pixel data (uint8)
        pixels = f.read(num_images * rows * cols)

        # Convert uint8 -> float32 and normalize to [0, 1]
        float_data = struct.pack(f"{len(pixels)}f", *[b / 255.0 for b in pixels])

        return Tensor.from_bytes(float_data, [num_images, 1, rows, cols])


def load_mnist_labels(filepath: str) -> Tensor:
    """Load MNIST labels from IDX file format."""
    with open(filepath, "rb") as f:
        magic = struct.unpack(">I", f.read(4))[0]
        assert magic == 2049, f"Invalid magic number: {magic}"

        num_labels = struct.unpack(">I", f.read(4))[0]
        labels = f.read(num_labels)

        # One-hot encode: (num_labels,) -> (num_labels, 10)
        one_hot = []
        for label in labels:
            row = [0.0] * 10
            row[label] = 1.0
            one_hot.extend(row)

        float_data = struct.pack(f"{len(one_hot)}f", *one_hot)
        return Tensor.from_bytes(float_data, [num_labels, 10])
