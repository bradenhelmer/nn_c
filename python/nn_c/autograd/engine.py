"""
nn_c.autograd.engine
~~~~~~~~~~~~~~~~~~~~
Autograd engine implementation
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from nn_c._nn_core import Tensor as _CTensor

if TYPE_CHECKING:
    from nn_c.tensor import Tensor


def backward(root: Tensor):
    """
    Run backward pass from this tensor.

    Assumes scalar loss, gradient == 1.
    """

    # 1. Init root gradient
    if root.grad is None:
        root.grad = _CTensor.ones_like(root._data)

    # 2. Topo sort to get reverse order
    topo_order = _topological_sort(root)

    # 3. Traverse in reverse to compute gradients
    for tensor in reversed(topo_order):
        if tensor.grad_fn is None or tensor.grad is None:
            continue  # Leaf or no gradient yet

        # Which gradients go to your inputs?
        grad_inputs = tensor.grad_fn.backward(tensor.grad)

        # Distribute gradients to parent tensors
        for parent, grad in zip(tensor.grad_fn.inputs, grad_inputs):
            if grad is None:
                continue  # Input doesn't need gradients

            # Accumulate gradient for shared parameters
            if parent.grad is None:
                # Import here to avoid circular dependency
                from nn_c.tensor import Tensor

                parent.grad = Tensor(grad, requires_grad=False)
            else:
                parent.grad._data = parent.grad._data.add(grad)


def _topological_sort(root: Tensor) -> list[Tensor]:
    """
    Returns tensors in topological order (depencies before dependents).

    Uses depth first search for ordering.
    """

    visited: set[int] = set()
    topo_order: list[Tensor] = []

    def dfs(tensor: Tensor) -> None:
        if id(tensor) in visited or tensor.grad_fn is None:
            return

        visited.add(id(tensor))

        # Visit parents first
        for parent in tensor.grad_fn.inputs:
            dfs(parent)

        topo_order.append(tensor)

    dfs(root)
    return topo_order
