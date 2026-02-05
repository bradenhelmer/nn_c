"""
nn_c.autograd.engine
~~~~~~~~~~~~~~~~~~~~
Backward pass execution for automatic differentiation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from nn_c.tensor import Tensor


def run_backward(root: Tensor) -> None:
    """
    Execute backward pass on the computation graph.

    Traverses the graph in reverse topological order, calling each node's
    grad_fn to compute and accumulate gradients.

    Parameters
    ----------
    root : Tensor
        Root tensor to backpropagate from. Must have grad initialized.
    """
    for tensor in reversed(_topological_sort(root)):
        if tensor.grad_fn is None or tensor.grad is None:
            continue

        grad_inputs = tensor.grad_fn(tensor.grad)

        for parent, grad in zip(tensor.inputs(), grad_inputs):
            if grad is None:
                continue

            if parent.grad is None:
                parent.grad = grad
            else:
                parent.grad = parent.grad.add(grad)


def _topological_sort(root: Tensor) -> list[Tensor]:
    """
    Sort computation graph in topological order.

    Parameters
    ----------
    root : Tensor
        Root of the computation graph.

    Returns
    -------
    list[Tensor]
        Tensors in topological order (dependencies before dependents).
    """
    visited: set[int] = set()
    order: list[Tensor] = []

    def dfs(tensor: Tensor) -> None:
        if id(tensor) in visited or tensor.grad_fn is None:
            return
        visited.add(id(tensor))
        for parent in tensor._inputs:
            dfs(parent)
        order.append(tensor)

    dfs(root)
    return order
