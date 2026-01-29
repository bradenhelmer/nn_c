"""
nn_c.tensor
~~~~~~~~~~~
Core tensor definition.
"""

from nn_c import autograd as ag
from nn_c._nn_core import Tensor as _CTensor


class Tensor:
    def __init__(self, data: "_CTensor | Tensor", requires_grad: bool = False):
        self._data: _CTensor
        self.grad_fn: ag.grad_function.GradFunction | None = None
        self.grad: _CTensor | None = None

        if isinstance(data, _CTensor):
            self._data = data
        else:
            self._data = data._data

        self.requires_grad: bool = requires_grad

    def backward(self):
        ag.engine.backward(self)

    def zero_grad(self):
        self.grad = None

    def add(self, other: "Tensor") -> "Tensor":
        return ag.ops.add(self, other)

    def matmul(self, other: "Tensor") -> "Tensor":
        return ag.ops.matmul(self, other)

    def relu(self) -> "Tensor":
        return ag.ops.relu(self)
