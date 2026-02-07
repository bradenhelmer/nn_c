"""
nn_c.optim.scheduler
~~~~~~~~~~~~~~~~~~~~
Learning rate schedulers.

Schedulers adjust the optimizer's learning rate during training.
Call `scheduler.step()` at the end of each epoch.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Protocol, final, runtime_checkable

if TYPE_CHECKING:
    from nn_c.optim import Optimizer


@runtime_checkable
class Scheduler(Protocol):
    """Protocol for all schedulers."""

    def step(self) -> None:
        """Update the learning rate. Call at the end of each epoch."""
        ...

    def get_lr(self) -> float:
        """Return the current learning rate."""
        ...


class _SchedulerBase:
    """Base class for schedulers with common functionality."""

    def __init__(self, optimizer: Optimizer) -> None:
        self.optimizer: Optimizer = optimizer
        self.initial_lr: float = optimizer.lr
        self.current_epoch: int = 0

    def get_lr(self) -> float:
        """Return the current learning rate."""
        return self.optimizer.lr

    def _set_lr(self, lr: float) -> None:
        """Set the optimizer's learning rate."""
        self.optimizer.lr = lr


@final
class ConstantScheduler(_SchedulerBase):
    """
    Constant learning rate (no decay).

    Useful as a baseline or when no scheduling is desired.

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose learning rate will be managed.
    """

    def __init__(self, optimizer: Optimizer) -> None:
        super().__init__(optimizer)

    def step(self) -> None:
        """No-op for constant scheduler."""
        self.current_epoch += 1


@final
class StepScheduler(_SchedulerBase):
    """
    Step decay scheduler.

    Decays learning rate by gamma every step_size epochs.

    lr = initial_lr * gamma^(epoch // step_size)

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose learning rate will be managed.
    step_size : int
        Decay learning rate every step_size epochs.
    gamma : float
        Multiplicative factor of learning rate decay. Default: 0.1.
    """

    def __init__(self, optimizer: Optimizer, step_size: int, gamma: float = 0.1) -> None:
        super().__init__(optimizer)
        self.step_size: int = step_size
        self.gamma: float = gamma

    def step(self) -> None:
        """Apply step decay."""
        self.current_epoch += 1
        new_lr = self.initial_lr * (self.gamma ** (self.current_epoch // self.step_size))
        self._set_lr(new_lr)


@final
class ExponentialScheduler(_SchedulerBase):
    """
    Exponential decay scheduler.

    lr = initial_lr * e^(-decay_rate * epoch)

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose learning rate will be managed.
    decay_rate : float
        Exponential decay rate.
    """

    def __init__(self, optimizer: Optimizer, decay_rate: float) -> None:
        super().__init__(optimizer)
        self.decay_rate = decay_rate

    def step(self) -> None:
        """Apply exponential decay."""
        self.current_epoch += 1
        new_lr = self.initial_lr * math.exp(-self.decay_rate * self.current_epoch)
        self._set_lr(new_lr)


@final
class CosineAnnealingScheduler(_SchedulerBase):
    """
    Cosine annealing scheduler.

    Smoothly decays learning rate from initial_lr to min_lr following a cosine curve.

    lr = min_lr + 0.5 * (initial_lr - min_lr) * (1 + cos(epoch * pi / T_max))

    Parameters
    ----------
    optimizer : Optimizer
        Optimizer whose learning rate will be managed.
    T_max : int
        Maximum number of epochs (period of the cosine).
    min_lr : float
        Minimum learning rate. Default: 0.0.
    """

    def __init__(self, optimizer: Optimizer, T_max: int, min_lr: float = 0.0) -> None:
        super().__init__(optimizer)
        self.T_max: int = T_max
        self.min_lr: float = min_lr

    def step(self) -> None:
        """Apply cosine annealing."""
        self.current_epoch += 1
        new_lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (
            1.0 + math.cos(self.current_epoch * math.pi / self.T_max)
        )
        self._set_lr(new_lr)
