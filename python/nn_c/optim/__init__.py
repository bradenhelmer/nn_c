"""
nn_c.optim
~~~~~~~~~~
Optimization algorithms and learning rate schedulers.
"""

from typing import Protocol, runtime_checkable

from nn_c.optim.adam import Adam
from nn_c.optim.momentum import Momentum
from nn_c.optim.scheduler import (
    ConstantScheduler,
    CosineAnnealingScheduler,
    ExponentialScheduler,
    Scheduler,
    StepScheduler,
)
from nn_c.optim.sgd import SGD


@runtime_checkable
class Optimizer(Protocol):
    """Protocol for all optimizers."""

    lr: float

    def step(self) -> None:
        """Apply one optimization step."""
        ...

    def zero_grad(self) -> None:
        """Zero out gradients for all parameters."""
        ...


__all__ = [
    "Adam",
    "ConstantScheduler",
    "CosineAnnealingScheduler",
    "ExponentialScheduler",
    "Momentum",
    "Optimizer",
    "Scheduler",
    "SGD",
    "StepScheduler",
]
