"""
nn_c.nn.linear
~~~~~~~~~~~~~~
Linear (fully connected) layer.
"""

import math
from collections.abc import Iterator
from typing import override

from nn_c import Tensor
from nn_c.nn.module import Module


class Linear(Module):
    """
    Fully connected layer: y = x @ W + b.

    Parameters
    ----------
    in_features : int
        Size of each input sample.
    out_features : int
        Size of each output sample.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        # Xavier/Glorot initialization: scale = sqrt(2 / (fan_in + fan_out))
        scale = math.sqrt(2.0 / (in_features + out_features))
        self.weights: Tensor = Tensor.random(
            [in_features, out_features], -scale, scale, requires_grad=True
        )
        self.biases: Tensor = Tensor([out_features], requires_grad=True)

    @override
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (batch_size, in_features).

        Returns
        -------
        Tensor
            Output tensor of shape (batch_size, out_features).
        """
        out = x.matmul(self.weights)
        out = out.add(self.biases)
        return out

    @override
    def parameters(self) -> Iterator[Tensor]:
        yield self.weights
        yield self.biases
