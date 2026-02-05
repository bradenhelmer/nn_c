"""
nn_c.dataset.dataloader
~~~~~~~~~~~~~~~~~~~~~~~
DataLoader and TensorDataset for batching samples.
"""

import random
from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from nn_c.tensor import Tensor


@runtime_checkable
class Dataset(Protocol):
    """
    Protocol for datasets.

    A dataset provides indexed access to samples. Each sample is a tuple of
    tensors (e.g., features and labels, or input/mask/label for segmentation).
    """

    def __len__(self) -> int:
        """Return the total number of samples."""
        ...

    def __getitem__(self, idx: int) -> tuple[Tensor, ...]:
        """
        Return the sample at the given index.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple[Tensor, ...]
            Tuple of tensors for this sample (e.g., features, labels).
        """
        ...


class TensorDataset:
    """
    Dataset wrapping feature and label tensors.

    This is a convenience class for the common case where you have two
    pre-loaded tensors (features and labels) and want to use them with
    a DataLoader. For custom data loading (lazy loading, augmentation, etc.),
    implement the Dataset protocol directly.

    Parameters
    ----------
    features : Tensor
        Feature tensor with shape (num_samples, ...).
    labels : Tensor
        Label tensor with shape (num_samples, ...).
    """

    def __init__(self, features: Tensor, labels: Tensor) -> None:
        if features.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Feature and label tensors must have same number of samples, "
                f"got {features.shape[0]} and {labels.shape[0]}"
            )

        self._features: Tensor = features
        self._labels: Tensor = labels

        self._feat_data: list[float] = features.to_list()
        self._feat_shape: list[int] = list(features.shape[1:])
        self._feat_stride: int = 1
        for d in self._feat_shape:
            self._feat_stride *= d

        self._label_data: list[float] = labels.to_list()
        self._label_shape: list[int] = list(labels.shape[1:])
        self._label_stride: int = 1
        for d in self._label_shape:
            self._label_stride *= d

    def __len__(self) -> int:
        """Return the number of samples."""
        return self._features.shape[0]

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        """
        Return the sample at the given index.

        Parameters
        ----------
        idx : int
            Sample index.

        Returns
        -------
        tuple[Tensor, Tensor]
            Feature and label tensors for this sample.
        """
        feat_start = idx * self._feat_stride
        feat_end = feat_start + self._feat_stride
        feat_data = self._feat_data[feat_start:feat_end]

        label_start = idx * self._label_stride
        label_end = label_start + self._label_stride
        label_data = self._label_data[label_start:label_end]

        return (
            Tensor(self._feat_shape, feat_data),
            Tensor(self._label_shape, label_data),
        )


class DataLoader:
    """
    Batches samples from a dataset.

    Iterates over a dataset, yielding batches of stacked tensors. Supports
    shuffling and dropping the last incomplete batch.

    Parameters
    ----------
    dataset : Dataset
        Dataset to load samples from.
    batch_size : int
        Number of samples per batch.
    shuffle : bool
        If True, shuffle samples each epoch.
    drop_last : bool
        If True, drop the last batch if it's smaller than batch_size.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        shuffle: bool = False,
        drop_last: bool = False,
    ) -> None:
        self.dataset: Dataset = dataset
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.drop_last: bool = drop_last

    def __len__(self) -> int:
        """Return the number of batches per epoch."""
        n = len(self.dataset)
        if self.drop_last:
            return n // self.batch_size
        return (n + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[Tensor, ...]]:
        """Iterate over batches."""
        indices = list(range(len(self.dataset)))
        if self.shuffle:
            random.shuffle(indices)

        for start in range(0, len(indices), self.batch_size):
            batch_indices = indices[start : start + self.batch_size]

            if self.drop_last and len(batch_indices) < self.batch_size:
                break

            samples = [self.dataset[i] for i in batch_indices]
            yield self._collate(samples)

    def _collate(self, samples: list[tuple[Tensor, ...]]) -> tuple[Tensor, ...]:
        """
        Stack samples into batched tensors.

        Parameters
        ----------
        samples : list[tuple[Tensor, ...]]
            List of samples, each a tuple of tensors.

        Returns
        -------
        tuple[Tensor, ...]
            Tuple of batched tensors, one per item in the sample tuple.
        """
        num_items = len(samples[0])
        batched: list[Tensor] = []

        for item_idx in range(num_items):
            items = [s[item_idx] for s in samples]
            batch_shape = [len(items)] + list(items[0].shape)
            batch_data: list[float] = []
            for item in items:
                batch_data.extend(item.to_list())
            batched.append(Tensor(batch_shape, batch_data))

        return tuple(batched)
