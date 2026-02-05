"""
nn_c.nn.module
~~~~~~~~~~~~~~
Base class for neural network modules.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from nn_c import Tensor


class Module(ABC):
    """Base class for all neural network modules."""

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor.
        """
        pass

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    @abstractmethod
    def parameters(self) -> Iterator[Tensor]:
        """
        Iterate over module parameters.

        Yields
        ------
        Tensor
            Each parameter tensor in the module.
        """
        pass
