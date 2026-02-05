"""
nn_c.dataset.mnist
~~~~~~~~~~~~~~~~~~
MNIST dataset loader.
"""

import struct
from pathlib import Path
from typing import cast

from nn_c import Tensor

__all__ = ["load_mnist"]


def _load_mnist_images(filepath: Path) -> Tensor:
    """Load MNIST images from IDX file format."""
    with open(filepath, "rb") as f:
        magic = cast(int, struct.unpack(">I", f.read(4))[0])
        if magic != 2051:
            raise ValueError(f"Invalid magic number: {magic}")

        num_images = cast(int, struct.unpack(">I", f.read(4))[0])
        rows = cast(int, struct.unpack(">I", f.read(4))[0])
        cols = cast(int, struct.unpack(">I", f.read(4))[0])

        pixels = f.read(num_images * rows * cols)
        float_data = struct.pack(f"{len(pixels)}f", *[b / 255.0 for b in pixels])

        return Tensor([num_images, rows * cols], float_data)


def _load_mnist_labels(filepath: Path) -> Tensor:
    """Load MNIST labels from IDX file format."""
    with open(filepath, "rb") as f:
        magic = cast(int, struct.unpack(">I", f.read(4))[0])
        if magic != 2049:
            raise ValueError(f"Invalid magic number: {magic}")

        num_labels = cast(int, struct.unpack(">I", f.read(4))[0])
        labels = f.read(num_labels)

        one_hot: list[float] = []
        for label in labels:
            row: list[float] = [0.0] * 10
            row[label] = 1.0
            one_hot.extend(row)

        float_data = struct.pack(f"{len(one_hot)}f", *one_hot)
        return Tensor([num_labels, 10], float_data)


def load_mnist(image_filepath: Path, label_filepath: Path) -> tuple[Tensor, Tensor]:
    """
    Load MNIST dataset from IDX files.

    Parameters
    ----------
    image_filepath : Path
        Path to the MNIST images file.
    label_filepath : Path
        Path to the MNIST labels file.

    Returns
    -------
    tuple[Tensor, Tensor]
        Images tensor of shape (N, 784) and labels tensor of shape (N, 10).
    """
    return _load_mnist_images(image_filepath), _load_mnist_labels(label_filepath)
