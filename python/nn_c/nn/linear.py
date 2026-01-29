"""
nn_core.nn.linear
~~~~~~~~~
Linear layer class
"""

from nn_c._nn_core import Tensor
from nn_c import _CTensor


class Linear:
    """
    Fully connected layer:

    y = xW^T + b
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        self.weights: Tensor = Tensor([in_features, out_features])
        self.biases: Tensor = Tensor([out_features])

    def forward(self, x: Tensor) -> Tensor:
        out = x.matmul(self.weights.transpose2d())
        out = out.add(self.biases)
        return out
