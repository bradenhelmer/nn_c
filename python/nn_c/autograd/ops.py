# pyright: reportPrivateUsage=false
"""
nn_c.autograd.ops
~~~~~~~~~~~~~~~~~
Autograd op definitions.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nn_c.autograd.function import AddBackward, MatMulBackward, ReLUBackward, Transpose2DBackward

if TYPE_CHECKING:
    from nn_c.tensor import Tensor


def add(a: Tensor, b: Tensor) -> Tensor:
    result_data = a._data.add(b._data)

    if a.requires_grad or b.requires_grad:
        grad_fn = AddBackward()
        grad_fn.inputs = [a, b]
        # Gradient doesn't depend on results, don't save tensors.

        result = Tensor(result_data, requires_grad=True)
        result.grad_fn = grad_fn
        return result

    return Tensor(result_data, requires_grad=False)


def relu(x: Tensor) -> Tensor:
    result_data = x._data.relu()

    if x.requires_grad:
        grad_fn = ReLUBackward()
        grad_fn.inputs = [x]
        grad_fn.save_for_backward(x._data)

        result = Tensor(result_data, requires_grad=True)
        result.grad_fn = grad_fn
        return result

    return Tensor(result_data, requires_grad=False)


def matmul(a: Tensor, b: Tensor) -> Tensor:
    result_data = a._data.matmul(b._data)

    if a.requires_grad or b.requires_grad:
        grad_fn = MatMulBackward()
        grad_fn.inputs = [a, b]
        grad_fn.save_for_backward(a._data, b._data)

        result = Tensor(result_data, requires_grad=True)
        result.grad_fn = grad_fn
        return result

    return Tensor(result_data, requires_grad=False)


def transpose2d(x: Tensor) -> Tensor:
    result_data = x._data.transpose2d()

    if x.requires_grad:
        grad_fn = Transpose2DBackward()
        grad_fn.inputs = [x]
        grad_fn.save_for_backward(x._data)

        result = Tensor(result_data, requires_grad=True)
        result.grad_fn = grad_fn
        return result

    return Tensor(result_data, requires_grad=False)
