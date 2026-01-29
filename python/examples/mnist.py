"""
examples.mnist
~~~~~~~~~~~~~~
MNIST Examples
"""

from pathlib import Path

from nn_c.dataset.mnist import load_mnist


def mnist_sgd():
    print("Traning MNIST with SGD optimizer...")

    mnist_train_imgs, mnist_train_labels = load_mnist(
        Path("datasets/mnist/train_imgs"), Path("datasets/mnist/train_labels")
    )

    print(f"MNIST Train Img Count: {mnist_train_imgs.shape[0]}")
    print(f"MNIST Train Label Count: {mnist_train_labels.shape[0]}")


if __name__ == "__main__":
    mnist_sgd()
