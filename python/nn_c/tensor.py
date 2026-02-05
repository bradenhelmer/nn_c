"""
nn_c.tensor
~~~~~~~~~~~
Core tensor class with integrated autograd.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import override

from nn_c._nn_core import Tensor as _CTensor

from nn_c.autograd.engine import run_backward


class Tensor:
    """N-dimensional tensor with automatic differentiation support."""

    __slots__: tuple[str, ...] = ("_data", "grad", "grad_fn", "_inputs", "requires_grad")

    def __init__(
        self,
        shape: list[int],
        data: bytes | list[float] | None = None,
        requires_grad: bool = False,
    ) -> None:
        self._data: _CTensor
        if isinstance(data, bytes):
            self._data = _CTensor.from_bytes(data, shape)
        elif isinstance(data, list):
            self._data = _CTensor.from_list(data, shape)
        else:
            self._data = _CTensor(shape)

        self.grad: Tensor | None = None
        self.grad_fn: Callable[[Tensor], tuple[Tensor | None, ...]] | None = None
        self._inputs: list[Tensor] = []
        self.requires_grad: bool = requires_grad

    # -------------------------------------------------------------------------
    # Internal Factory
    # -------------------------------------------------------------------------

    @classmethod
    def _from_ctensor(cls, data: _CTensor, requires_grad: bool = False) -> Tensor:
        """Create tensor from raw C tensor (internal use only)."""
        t = cls.__new__(cls)
        t._data = data
        t.grad = None
        t.grad_fn = None
        t._inputs = []
        t.requires_grad = requires_grad
        return t

    # -------------------------------------------------------------------------
    # Static Factories
    # -------------------------------------------------------------------------

    @staticmethod
    def from_list(data: list[float], shape: list[int]) -> Tensor:
        """Create tensor from a flat list of floats."""
        return Tensor(shape, data=data)

    @staticmethod
    def from_bytes(data: bytes, shape: list[int]) -> Tensor:
        """Create tensor from raw bytes."""
        return Tensor(shape, data=data)

    @staticmethod
    def ones_like(other: Tensor) -> Tensor:
        """Create tensor of ones with same shape as other."""
        return Tensor._from_ctensor(_CTensor.ones_like(other._data))

    @staticmethod
    def zeros_like(other: Tensor) -> Tensor:
        """Create tensor of zeros with same shape as other."""
        return Tensor(list(other.shape))

    @staticmethod
    def random(
        shape: list[int], min_val: float, max_val: float, requires_grad: bool = False
    ) -> Tensor:
        """Create tensor with uniform random values in [min_val, max_val]."""
        return Tensor._from_ctensor(_CTensor.random(shape, min_val, max_val), requires_grad)

    # -------------------------------------------------------------------------
    # Properties
    # -------------------------------------------------------------------------

    @property
    def size(self) -> int:
        """Total number of elements."""
        return self._data.size

    @property
    def ndim(self) -> int:
        """Number of dimensions."""
        return self._data.ndim

    @property
    def shape(self) -> tuple[int, ...]:
        """Shape of the tensor."""
        return self._data.shape

    # -------------------------------------------------------------------------
    # Autograd
    # -------------------------------------------------------------------------

    def backward(self) -> None:
        """Run backward pass from this tensor."""
        if self.grad is None:
            self.grad = Tensor.ones_like(self)

        run_backward(self)

    def zero_grad(self) -> None:
        """Clear gradient."""
        self.grad = None

    def sub_inplace(self, other: Tensor, scale: float = 1.0) -> None:
        """
        In-place subtraction: self = self - scale * other.

        Parameters
        ----------
        other : Tensor
            Tensor to subtract.
        scale : float
            Scaling factor applied to other before subtraction.
        """
        self._data = self._data.subtract(other._data.scale(scale))

    # -------------------------------------------------------------------------
    # Operations
    # -------------------------------------------------------------------------

    def add(self, other: Tensor) -> Tensor:
        """Element-wise addition."""
        result = Tensor._from_ctensor(self._data.add(other._data))

        if self.requires_grad or other.requires_grad:
            saved_self_shape = self.shape
            saved_other_shape = other.shape

            def backward(grad: Tensor) -> tuple[Tensor, Tensor]:
                grad_self = grad
                grad_other = grad
                # Handle broadcasting: sum over broadcast dimensions
                if saved_self_shape != saved_other_shape:
                    # Case: other is 1D, self is 2D (bias add)
                    if len(saved_other_shape) == 1 and len(saved_self_shape) == 2:
                        # Sum over batch dimension (axis 0)
                        grad_other = grad.sum_axis(0)
                return (grad_self, grad_other)

            result.requires_grad = True
            result._inputs = [self, other]
            result.grad_fn = backward

        return result

    def matmul(self, other: Tensor) -> Tensor:
        """Matrix multiplication."""
        result = Tensor._from_ctensor(self._data.matmul(other._data))

        if self.requires_grad or other.requires_grad:
            saved_self, saved_other = self, other

            def backward(grad: Tensor) -> tuple[Tensor, Tensor]:
                grad_self = grad.matmul(saved_other.transpose2d())
                grad_other = saved_self.transpose2d().matmul(grad)
                return (grad_self, grad_other)

            result.requires_grad = True
            result._inputs = [self, other]
            result.grad_fn = backward

        return result

    def relu(self) -> Tensor:
        """ReLU activation."""
        result = Tensor._from_ctensor(self._data.relu())

        if self.requires_grad:
            saved_result = result

            def backward(grad: Tensor) -> tuple[Tensor]:
                grad_input = Tensor._from_ctensor(grad._data.relu_backward(saved_result._data))
                return (grad_input,)

            result.requires_grad = True
            result._inputs = [self]
            result.grad_fn = backward

        return result

    def transpose2d(self) -> Tensor:
        """Transpose 2D tensor."""
        result = Tensor._from_ctensor(self._data.transpose2d())

        if self.requires_grad:

            def backward(grad: Tensor) -> tuple[Tensor]:
                return (grad.transpose2d(),)

            result.requires_grad = True
            result._inputs = [self]
            result.grad_fn = backward

        return result

    def softmax_cross_entropy(self, target: Tensor) -> Tensor:
        """Softmax cross entropy loss."""
        loss_val: float = self._data.softmax_cross_entropy(target._data)
        result = Tensor([1], data=[loss_val])

        if self.requires_grad:
            saved_self, saved_target = self, target

            def backward(grad: Tensor) -> tuple[Tensor]:
                grad_logits = Tensor._from_ctensor(
                    saved_self._data.softmax_cross_entropy_backward(saved_target._data)
                )
                return (grad_logits,)

            result.requires_grad = True
            result._inputs = [self]
            result.grad_fn = backward

        return result

    def sum_axis(self, axis: int) -> Tensor:
        """Sum tensor along specified axis."""
        return Tensor._from_ctensor(self._data.sum_axis(axis))

    # -------------------------------------------------------------------------
    # Data Access
    # -------------------------------------------------------------------------

    def __getitem__(self, idx: int | tuple[int, ...]) -> float:
        return self._data[idx]

    def __setitem__(self, idx: int | tuple[int, ...], value: float) -> None:
        self._data[idx] = value

    def __len__(self) -> int:
        return len(self._data)

    @override
    def __repr__(self) -> str:
        grad_info = ", requires_grad=True" if self.requires_grad else ""
        return f"Tensor(shape={list(self.shape)}{grad_info})"

    def to_list(self) -> list[float]:
        """Export tensor data as flat list."""
        return self._data.to_list()

    def to_bytes(self) -> bytes:
        """Export tensor data as raw bytes."""
        return self._data.to_bytes()

    def inputs(self) -> list[Tensor]:
        """Return tensor inputs for backward pass."""
        return self._inputs
