"""
nn_c.optim.adam
~~~~~~~~~~~~~~~
Adam optimizer.
"""

from collections.abc import Iterator

from nn_c.tensor import Tensor


class Adam:
    """
    Adam optimizer.

    Updates parameters using:

    m = beta1 * m + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * grad^2
    m_hat = m / (1 - beta1^t)
    s_hat = s / (1 - beta2^t)
    param = param - lr * m_hat / (sqrt(s_hat) + epsilon)

    Parameters
    ----------
    params : Iterator[Tensor]
        Iterator over parameters to optimize.
    lr : float
        Learning rate.
    beta1 : float
        Exponential decay rate for first moment estimates.
    beta2 : float
        Exponential decay rate for second moment estimates.
    epsilon : float
        Small constant for numerical stability.
    """

    def __init__(
        self,
        params: Iterator[Tensor],
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ) -> None:
        self.params: list[Tensor] = list(params)
        self.lr: float = lr
        self.beta1: float = beta1
        self.beta2: float = beta2
        self.epsilon: float = epsilon
        self.timestep: int = 0
        self.m: list[Tensor] = []
        self.s: list[Tensor] = []
        self._init_moment_tensors()

    def _init_moment_tensors(self) -> None:
        """Initialize first and second moment tensors to zeros."""
        for param in self.params:
            self.m.append(Tensor.zeros_like(param))
            self.s.append(Tensor.zeros_like(param))

    def step(self) -> None:
        """Apply one optimization step."""
        self.timestep += 1

        # Precompute bias corrections
        bc1 = 1.0 - self.beta1**self.timestep
        bc2 = 1.0 - self.beta2**self.timestep

        for i, param in enumerate(self.params):
            if param.grad is None:
                continue

            grad = param.grad

            # m = beta1 * m + (1 - beta1) * grad
            self.m[i] = self.m[i].scale(self.beta1).add(grad.scale(1.0 - self.beta1))

            # s = beta2 * s + (1 - beta2) * grad^2
            self.s[i] = (
                self.s[i].scale(self.beta2).add(grad.elementwise_mul(grad).scale(1.0 - self.beta2))
            )

            # m_hat = m / (1 - beta1^t)
            m_hat = self.m[i].scale(1.0 / bc1)

            # s_hat = s / (1 - beta2^t)
            s_hat = self.s[i].scale(1.0 / bc2)

            # param -= lr * m_hat / (sqrt(s_hat) + epsilon)
            update = m_hat.elementwise_div(s_hat.sqrt().add_scalar(self.epsilon))
            param.sub_inplace(update, self.lr)

    def zero_grad(self) -> None:
        """Zero out gradients for all parameters."""
        for param in self.params:
            param.zero_grad()
