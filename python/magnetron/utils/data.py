# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import math
import random
from abc import ABC, abstractmethod

from magnetron import Tensor, dtype
from typing import Generic, TypeVar, Iterator, override, Any, Mapping, Sequence, Callable

T = TypeVar('T')


class Dataset(ABC, Generic[T]):
    @abstractmethod
    def __getitem__(self, i: int) -> T: ...

    @abstractmethod
    def __len__(self) -> int: ...

    def __getitems__(self, idx: list[int]) -> list[T]:
        return [self[i] for i in idx]


class TensorDataset(Dataset[tuple[Tensor, ...]]):
    def __init__(self, *tensors: Tensor) -> None:
        if not tensors:
            raise ValueError('TensorDataset needs at least one tensor')
        n = tensors[0].shape[0]
        if any(t.shape[0] != n for t in tensors):
            raise ValueError('All tensors must have the same size in dim 0')
        self.tensors: tuple[Tensor, ...] = tensors

    @override
    def __getitem__(self, i: int) -> tuple[Tensor, ...]:
        return tuple(t[i] for t in self.tensors)

    @override
    def __len__(self) -> int:
        return self.tensors[0].shape[0]

    @override
    def __getitems__(self, idx: list[int]) -> list[T]:
        idx = Tensor(idx, dtype=dtype.int64)
        return tuple(t[idx] for t in self.tensors)


class Sampler(ABC):
    @abstractmethod
    def __iter__(self) -> Iterator[int]: ...

    @abstractmethod
    def __len__(self) -> int: ...


class SequentialSampler(Sampler):
    def __init__(self, n: int) -> None:
        self.n: int = n

    @override
    def __iter__(self) -> Iterator[int]:
        return iter(range(self.n))

    @override
    def __len__(self) -> int:
        return self.n


class RandomSampler(Sampler):
    def __init__(self, n: int, generator: random.Random | None = None) -> None:
        self.n: int = n
        self.gen = generator or random.Random()

    @override
    def __iter__(self) -> Iterator[int]:
        perm = list(range(self.n))
        self.gen.shuffle(perm)
        return iter(perm)

    @override
    def __len__(self) -> int:
        return self.n


class BatchSampler(Sampler):
    def __init__(self, sampler: Sampler, batch_size: int, drop_last: bool) -> None:
        if batch_size <= 0:
            raise ValueError('batch_size must be positive')
        self.sampler = sampler
        self.batch_size = batch_size
        self.drop_last = drop_last

    @override
    def __iter__(self) -> Iterator[list[int]]:
        batch: list[int] = []
        for i in self.sampler:
            batch.append(i)
            if len(batch) == self.batch_size:
                yield batch
                batch = []
        if batch and not self.drop_last:
            yield batch

    @override
    def __len__(self) -> int:
        n = len(self.sampler)
        return n // self.batch_size if self.drop_last else math.ceil(n / self.batch_size)


def default_collate(batch: list[Any]) -> Any:
    elem = batch[0]
    if isinstance(elem, Tensor):
        return Tensor.stack(batch)
    if isinstance(elem, bool):
        return Tensor(batch, dtype=dtype.boolean)
    if isinstance(elem, int):
        return Tensor(batch, dtype=dtype.int64)
    if isinstance(elem, float):
        return Tensor(batch, dtype=dtype.float32)
    if isinstance(elem, Mapping):
        return {k: default_collate([b[k] for b in batch]) for k in elem}
    if isinstance(elem, tuple) and hasattr(elem, '_fields'):
        return type(elem)(*(default_collate(list(s)) for s in zip(*batch)))
    if isinstance(elem, Sequence) and not isinstance(elem, str):
        return type(elem)(default_collate(list(s)) for s in zip(*batch))
    return batch


class DataLoader(Generic[T]):
    def __init__(
        self,
        dataset: Dataset[T],
        batch_size: int = 1,
        shuffle: bool = False,
        drop_last: bool = False,
        sampler: Sampler | None = None,
        batch_sampler: BatchSampler | None = None,
        collate_fn: Callable[[list[T]], Any] | None = None,
        seed: int | None = None,
    ) -> None:
        if sampler is not None and shuffle:
            raise ValueError('sampler and shuffle are mutually exclusive')
        if batch_sampler is not None and (sampler is not None or shuffle or drop_last or batch_size != 1):
            raise ValueError('batch_sampler is mutually exclusive with batch_size, shuffle, sampler, drop_last')
        if batch_sampler is None:
            if sampler is None:
                n = len(dataset)
                sampler = RandomSampler(n, random.Random(seed)) if shuffle else SequentialSampler(n)
        batch_sampler = BatchSampler(sampler, batch_size, drop_last)
        self.dataset = dataset
        self.batch_sampler = batch_sampler
        self.collate_fn = collate_fn
        self._use_getitems = collate_fn is None and type(dataset).__getitems__ is not Dataset.__getitems__
        self.collate_fn = collate_fn or default_collate

    def __iter__(self) -> Iterator[Any]:
        for indices in self.batch_sampler:
            if self._use_getitems:
                yield self.dataset.__getitems__(indices)
            else:
                yield self.collate_fn([self.dataset[i] for i in indices])

    def __len__(self) -> int:
        return len(self.batch_sampler)
