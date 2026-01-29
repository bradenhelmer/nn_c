"""
nn_c.autograd.grad_function
~~~~~~~~~~~~~~~~~~~~~~~~~~~
GradFunction class definition.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, override

from nn_c._nn_core import Tensor as _CTensor

if TYPE_CHECKING:
    from nn_c.tensor import Tensor


class Function(ABC):
    """Base class for all gradient functions."""

    def __init__(self):
        self.saved_tensors: list[_CTensor] = []
        self.inputs: list[Tensor] = []

    def save_for_backward(self, *tensors: _CTensor):
        self.saved_tensors = list(tensors)

    @abstractmethod
    def backward(self, grad_output: _CTensor) -> tuple[_CTensor, ...] | None:
        pass


class AddBackward(Function):
    """Gradient for c = a + b"""

    @override
    def backward(self, grad_output: _CTensor) -> tuple[_CTensor, ...]:
        # d(a + b) / da = 1, d(a + b) / db = 1
        return (grad_output, grad_output)


class MatMulBackward(Function):
    """Gradient for C = A @ B"""

    @override
    def backward(self, grad_output: _CTensor) -> tuple[_CTensor, ...]:
        A, B = self.saved_tensors

        # dL/dA = dL/dC @ B^T
        grad_A: _CTensor = grad_output.matmul(B.transpose2d())

        # dL/dB = A^T @ dL/dC
        grad_B: _CTensor = A.transpose2d().matmul(grad_output)

        return (grad_A, grad_B)


class ReLUBackward(Function):
    """Gradient for ReLU"""

    @override
    def backward(self, grad_output: _CTensor) -> tuple[_CTensor, ...]:
        input_tensor: _CTensor = self.saved_tensors[0]
        grad_input = _CTensor.relu_backward(grad_output, input_tensor)

        return (grad_input,)
