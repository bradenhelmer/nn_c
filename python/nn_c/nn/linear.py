"""
nn_core.nn.linear
~~~~~~~~~
Linear layer class
"""

from collections.abc import Iterator
from typing import override

from nn_c import Tensor
from nn_c.nn.module import Module


class Linear(Module):
    """
    Fully connected layer:

    y = xW^T + b
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        self.weights: Tensor = Tensor([in_features, out_features])
        self.biases: Tensor = Tensor([out_features])

    @override
    def forward(self, x: Tensor) -> Tensor:
        out = x.matmul(self.weights.transpose2d())
        out = out.add(self.biases)
        return out

    @override
    def parameters(self) -> Iterator[Tensor]:
        yield self.weights
        yield self.biases
