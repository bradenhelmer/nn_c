"""
nn_c.optim
~~~~~~~~~~
Optimization algorithms.
"""

from typing import Protocol, runtime_checkable

from nn_c.optim.sgd import SGD


@runtime_checkable
class Optimizer(Protocol):
    """Protocol for all optimizers."""

    def step(self) -> None:
        """Apply one optimization step."""
        ...

    def zero_grad(self) -> None:
        """Zero out gradients for all parameters."""
        ...


__all__ = ["Optimizer", "SGD"]
