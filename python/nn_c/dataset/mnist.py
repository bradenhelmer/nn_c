"""
nn_c.dataset.mnist
~~~~~~~~~~~~~~~~~~
MNIST dataset loaders.
"""

import struct
from pathlib import Path
from typing import cast

from nn_c import Tensor

__all__ = ["load_mnist"]


def _load_mnist_images(filepath: Path) -> Tensor:
    """Load MNIST images from IDX file format."""
    with open(filepath, "rb") as mnist_file:
        magic = cast(int, struct.unpack(">I", mnist_file.read(4))[0])

        if magic != 2051:
            raise ValueError(f"Invalid magic number: {magic}")

        num_images = cast(int, struct.unpack(">I", mnist_file.read(4))[0])
        rows = cast(int, struct.unpack(">I", mnist_file.read(4))[0])
        cols = cast(int, struct.unpack(">I", mnist_file.read(4))[0])

        pixels = mnist_file.read(num_images * rows * cols)

        float_data = struct.pack(f"{len(pixels)}f", *[b / 255.0 for b in pixels])

        return Tensor([num_images, 1, rows, cols], float_data)


def _load_mnist_labels(filepath: Path) -> Tensor:
    """Load MNIST labels from IDX file format."""
    with open(filepath, "rb") as mnist_file:
        magic: int = cast(int, struct.unpack(">I", mnist_file.read(4))[0])

        if magic != 2049:
            raise ValueError(f"Invalid magic number: {magic}")

        num_labels: int = cast(int, struct.unpack(">I", mnist_file.read(4))[0])
        labels = mnist_file.read(num_labels)

        # One-hot encode (num_labels,) -> (num_labels, 10)
        one_hot: list[float] = []
        for label in labels:
            row: list[float] = [0.0] * 10
            row[label] = 1.0
            one_hot.extend(row)

        float_data = struct.pack(f"{len(one_hot)}f", *one_hot)
        return Tensor([num_labels, 10], float_data)


def load_mnist(image_filepath: Path, label_filepath: Path) -> tuple[Tensor, Tensor]:
    return _load_mnist_images(image_filepath), _load_mnist_labels(label_filepath)
