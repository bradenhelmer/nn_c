"""
nn_c.nn.module
~~~~~~~~~~~~~~
Neural net module base class defintion.
"""

from abc import ABC, abstractmethod
from collections.abc import Iterator

from nn_c import Tensor


class Module(ABC):
    """Base class for all layers."""

    @abstractmethod
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass method.

        Parameters
        ----------
        x : Tensor
            Input tensor.

        Returns
        -------
        Tensor
            Output tensor, or y.
        """
        pass

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    @abstractmethod
    def parameters(self) -> Iterator[Tensor]:
        """
        Gets the parameters associated with the layer.

        Returns
        -------
        Iterator[Tensor]
            An iterator over the module's parameters.
        """
        pass
