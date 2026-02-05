"""
nn_c.trainer
~~~~~~~~~~~~
Training loop with overridable hooks.
"""

from collections.abc import Callable

from nn_c.dataset.dataloader import DataLoader
from nn_c.nn.module import Module
from nn_c.optim import Optimizer
from nn_c.tensor import Tensor


class Trainer:
    """
    Training loop with overridable hooks.

    Provides a standard training loop that can be customized by overriding
    hook methods or the train/train_step methods entirely.

    Parameters
    ----------
    model : Module
        Neural network module to train.
    optimizer : Optimizer
        Optimizer for parameter updates.
    loss_fn : Callable[[Tensor, Tensor], Tensor]
        Loss function taking (predictions, targets) and returning scalar loss.
    dataloader : DataLoader
        DataLoader providing training batches.
    """

    def __init__(
        self,
        model: Module,
        optimizer: Optimizer,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        dataloader: DataLoader,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.dataloader = dataloader

    def train(self, epochs: int = 1) -> None:
        """
        Run the training loop.

        Override this method for full control over the training process.

        Parameters
        ----------
        epochs : int
            Number of epochs to train.
        """
        for epoch in range(epochs):
            self.on_epoch_start(epoch)
            epoch_loss = 0.0

            for batch_idx, batch in enumerate(self.dataloader):
                batch_x, batch_y = batch[0], batch[1]
                loss = self.train_step(batch_x, batch_y)
                epoch_loss += loss
                self.on_batch_end(batch_idx, loss)

            avg_loss = epoch_loss / len(self.dataloader) if len(self.dataloader) > 0 else 0.0
            self.on_epoch_end(epoch, avg_loss)

    def train_step(self, x: Tensor, y: Tensor) -> float:
        """
        Execute a single training step.

        Override for custom forward/backward logic.

        Parameters
        ----------
        x : Tensor
            Input batch.
        y : Tensor
            Target batch.

        Returns
        -------
        float
            Loss value for this batch.
        """
        logits = self.model(x)
        loss = self.loss_fn(logits, y)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss._data[0]

    def on_epoch_start(self, epoch: int) -> None:
        """
        Called at the start of each epoch.

        Override for custom behavior (e.g., learning rate scheduling).

        Parameters
        ----------
        epoch : int
            Current epoch index (0-based).
        """
        pass

    def on_epoch_end(self, epoch: int, avg_loss: float) -> None:
        """
        Called at the end of each epoch.

        Override for custom behavior (e.g., validation, checkpointing).

        Parameters
        ----------
        epoch : int
            Current epoch index (0-based).
        avg_loss : float
            Average loss for this epoch.
        """
        print(f"Epoch {epoch + 1} | loss: {avg_loss:.4f}")

    def on_batch_end(self, batch_idx: int, loss: float) -> None:
        """
        Called after each batch.

        Override for custom behavior (e.g., logging, gradient clipping).

        Parameters
        ----------
        batch_idx : int
            Current batch index within the epoch.
        loss : float
            Loss value for this batch.
        """
        pass
