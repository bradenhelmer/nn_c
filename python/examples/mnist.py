"""
examples.mnist
~~~~~~~~~~~~~~
MNIST Examples
"""

from collections.abc import Iterator
from pathlib import Path
from typing import override

from nn_c import Tensor
from nn_c.dataset.mnist import load_mnist
from nn_c.nn.linear import Linear
from nn_c.nn.module import Module


def mnist_sgd():
    print("Traning MNIST with SGD optimizer...")

    mnist_train_imgs, mnist_train_labels = load_mnist(
        Path("datasets/mnist/train_imgs"), Path("datasets/mnist/train_labels")
    )

    print(f"MNIST Train Img Count: {mnist_train_imgs.shape[0]}")
    print(f"MNIST Train Label Count: {mnist_train_labels.shape[0]}")

    class MnistSGD(Module):
        def __init__(self):
            self.linear_1: Linear = Linear(784, 128)
            self.linear_2: Linear = Linear(128, 10)

        @override
        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_1(x).relu()
            x = self.linear_2(x).relu()
            return x

        @override
        def parameters(self) -> Iterator[Tensor]:
            for param in self.linear_1.parameters():
                yield param
            for param in self.linear_2.parameters():
                yield param


if __name__ == "__main__":
    mnist_sgd()
