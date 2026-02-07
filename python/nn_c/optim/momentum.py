"""
nn_c.optim.momentum
~~~~~~~~~~~~~~
Momentum optimizer.
"""

from collections.abc import Iterator

from nn_c.tensor import Tensor


class Momentum:
    """
    Momentum optimizer.

    Updates parameters using:

    v = beta * v + grad
    param = param - lr * v

    Parameters
    ----------
    params : Iterator[Tensor]
        Iterator over parameters to optimize.
    lr : float
        Learning rate.
    beta : float
        Beta value.
    """

    def __init__(self, params: Iterator[Tensor], lr: float = 0.01, beta: float = 0.9) -> None:
        self.params: list[Tensor] = list(params)
        self.lr: float = lr
        self.beta: float = beta
        self.velocities: list[Tensor] = []
        self._init_velocity_tensors()

    def _init_velocity_tensors(self) -> None:
        """Initializes velocity tensors from params."""
        for param in self.params:
            self.velocities.append(Tensor.zeros_like(param))

    def step(self) -> None:
        """Apply one optimization step."""
        for i, (param, velocity) in enumerate(zip(self.params, self.velocities)):
            if param.grad is None:
                continue
            # v = beta * v + grad
            velocity = velocity.scale(self.beta).add(param.grad)
            self.velocities[i] = velocity
            # param = param - lr * v
            param.sub_inplace(velocity, self.lr)

    def zero_grad(self) -> None:
        """Zero out gradients for all parameters."""
        for param in self.params:
            param.zero_grad()
