"""
nn_c.tensor
~~~~~~~~~~~
Core tensor definition.
"""

from __future__ import annotations

from nn_c import autograd as ag
from nn_c._nn_core import Tensor as _CTensor


class Tensor:
    def __init__(self, data: _CTensor | Tensor, requires_grad: bool = False):
        self._data: _CTensor
        self.grad_fn: ag.function.Function | None = None
        self.grad: _CTensor | None = None

        if isinstance(data, _CTensor):
            self._data = data
        else:
            self._data = data._data

        self.requires_grad: bool = requires_grad

    @property
    def size(self) -> int:
        return self._data.size

    @property
    def ndim(self) -> int:
        return self._data.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        return self._data.shape

    def backward(self):
        ag.engine.backward(self)

    def zero_grad(self):
        self.grad = None

    def add(self, other: Tensor) -> Tensor:
        return ag.ops.add(self, other)

    def matmul(self, other: Tensor) -> Tensor:
        return ag.ops.matmul(self, other)

    def relu(self) -> Tensor:
        return ag.ops.relu(self)
