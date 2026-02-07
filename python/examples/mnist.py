"""
nn_c.examples.mnist
~~~~~~~~~~~~~~~~~~~
MNIST training example.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import final, override

from nn_c import Tensor
from nn_c.dataset import DataLoader, TensorDataset, load_mnist
from nn_c.dataset.mnist import mnist_comparator
from nn_c.nn.linear import Linear
from nn_c.nn.module import Module
from nn_c.optim import SGD, Optimizer
from nn_c.optim.adam import Adam
from nn_c.optim.momentum import Momentum
from nn_c.trainer import Trainer

IMAGES, LABELS = load_mnist(
    Path("datasets/mnist/train_imgs"),
    Path("datasets/mnist/train_labels"),
)


def test_mnist(model: Module) -> None:
    images, labels = load_mnist(
        Path("datasets/mnist/test_imgs"),
        Path("datasets/mnist/test_labels"),
    )
    test_dataset = TensorDataset(images, labels)
    correct = 0
    for image, label in test_dataset:
        # Reshape from (784,) to (1, 784) for batch dimension
        image_batch = image.reshape([1, 784])
        prediction = model.forward(image_batch)
        correct += mnist_comparator(label, prediction)

    print(
        f"Correct: {correct} / {len(test_dataset)}. Accuracy: {(correct / len(test_dataset)) * 100:.3f}%"
    )


@final
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


def loss_fn(logits: Tensor, target: Tensor) -> Tensor:
    return logits.softmax_cross_entropy(target)

def train_mnist_adam() -> None:
    print(f"Training MNIST with Adam optimizer.")
    model = MnistMLP()
    optimizer = Adam(model.parameters())
    dataset = TensorDataset(IMAGES, LABELS)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    trainer = Trainer(model, optimizer, loss_fn, dataloader)
    trainer.train(epochs=10)
    test_mnist(model)


if __name__ == "__main__":
    train_mnist_adam()
