"""
nn_c.optim.sgd
~~~~~~~~~~~~~~
Stochastic Gradient Descent optimizer.
"""

from collections.abc import Iterator

from nn_c.tensor import Tensor


class SGD:
    """
    Stochastic Gradient Descent optimizer.

    Updates parameters using: param = param - lr * grad

    Parameters
    ----------
    params : Iterator[Tensor]
        Iterator over parameters to optimize.
    lr : float
        Learning rate.
    """

    def __init__(self, params: Iterator[Tensor], lr: float = 0.01) -> None:
        self.params: list[Tensor] = list(params)
        self.lr = lr

    def step(self) -> None:
        """Apply one SGD update to all parameters."""
        for param in self.params:
            if param.grad is None:
                continue
            param.sub_inplace(param.grad, self.lr)

    def zero_grad(self) -> None:
        """Zero out gradients for all parameters."""
        for param in self.params:
            param.zero_grad()
