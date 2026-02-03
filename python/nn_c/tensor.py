"""
nn_c.tensor
~~~~~~~~~~~
Core tensor definition.
"""

from __future__ import annotations

from nn_c import autograd as ag
from nn_c._nn_core import Tensor as _CTensor


class Tensor:
    def __init__(
        self,
        shape: list[int],
        data: bytes | list[float] | _CTensor | None = None,
        requires_grad: bool = False,
    ):
        self._data: _CTensor
        if isinstance(data, bytes):
            self._data = _CTensor.from_bytes(data, shape)
        elif isinstance(data, list) and all(isinstance(x, float) for x in data):
            self._data = _CTensor.from_list(data, shape)
        elif isinstance(data, _CTensor):
            self._data = data
        elif data is None:
            self._data = _CTensor(shape)

        self.grad_fn: ag.function.Function | None = None
        self.grad: _CTensor | None = None
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

    def transpose2d(self) -> Tensor:
        return ag.ops.transpose2d(self)
