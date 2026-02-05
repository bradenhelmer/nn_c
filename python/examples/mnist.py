"""
nn_c.examples.mnist
~~~~~~~~~~~~~~~~~~~
MNIST training example.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import override

from nn_c import Tensor
from nn_c.dataset import DataLoader, TensorDataset, load_mnist
from nn_c.nn.linear import Linear
from nn_c.nn.module import Module
from nn_c.optim import SGD
from nn_c.trainer import Trainer


def mnist_sgd() -> None:
    """Train a simple MLP on MNIST using SGD."""
    print("Training MNIST with SGD optimizer...")

    images, labels = load_mnist(
        Path("datasets/mnist/train_imgs"),
        Path("datasets/mnist/train_labels"),
    )

    print(f"MNIST Train Images: {images.shape[0]}")
    print(f"MNIST Train Labels: {labels.shape[0]}")

    class MnistMLP(Module):
        def __init__(self) -> None:
            self.linear_1 = Linear(784, 128)
            self.linear_2 = Linear(128, 10)

        @override
        def forward(self, x: Tensor) -> Tensor:
            x = self.linear_1(x).relu()
            x = self.linear_2(x)
            return x

        @override
        def parameters(self) -> Iterator[Tensor]:
            yield from self.linear_1.parameters()
            yield from self.linear_2.parameters()

    model = MnistMLP()
    optimizer = SGD(model.parameters(), lr=0.01)
    dataset = TensorDataset(images, labels)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

    def loss_fn(logits: Tensor, target: Tensor) -> Tensor:
        return logits.softmax_cross_entropy(target)

    trainer = Trainer(model, optimizer, loss_fn, dataloader)
    trainer.train(epochs=10)


if __name__ == "__main__":
    mnist_sgd()
