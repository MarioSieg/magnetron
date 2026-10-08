# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

from magnetron import Tensor, context, nn
from magnetron.snapshot import SnapshotWriter, deserialize

CORRUPTIONS: tuple[str, ...] = ('noise', 'mask', 'both')


def preprocess(images: Tensor) -> Tensor:
    return images.cast(context.get_default_dtype()) / 255


def add_noise(x: Tensor, std: float) -> Tensor:
    return (x + Tensor.normal(*x.shape, mean=0.0, std=std)).clamp(0.0, 1.0)


def cut_holes(x: Tensor, size: int) -> Tensor:
    h, w = x.shape[-2:]
    holes = []
    for _ in range(x.shape[0]):
        top = random.randint(0, h - size)
        left = random.randint(0, w - size)
        holes.append(Tensor.ones(1, size, size).pad((left, w - size - left, top, h - size - top)))
    return x * (1.0 - Tensor.stack(holes))


def corrupt(x: Tensor, kind: str, noise_std: float, hole_size: int) -> Tensor:
    if kind not in CORRUPTIONS:
        raise ValueError(f'corruption must be one of {CORRUPTIONS}, but got {kind!r}')
    if kind in ('noise', 'both'):
        x = add_noise(x, noise_std)
    if kind in ('mask', 'both'):
        x = cut_holes(x, hole_size)
    return x


def _down(in_channels: int, out_channels: int, stride: int) -> list[nn.Module]:
    return [nn.Conv2D(in_channels, out_channels, 3, stride=stride, padding=1, bias=False), nn.GroupNorm(8, out_channels), nn.ReLU()]


def _up(in_channels: int, out_channels: int) -> list[nn.Module]:
    return [nn.ConvT2D(in_channels, out_channels, 4, stride=2, padding=1, bias=False), nn.GroupNorm(8, out_channels), nn.ReLU()]


class DenoisingAE(nn.Module):
    def __init__(self, image_size: int = 64, width: int = 32) -> None:
        super().__init__()
        if image_size % 8 != 0 or width % 8 != 0:
            raise ValueError(f'image_size and width must be multiples of 8, but got {image_size} and {width}')
        self.image_size = image_size
        self.width = width
        self.encoder = nn.Sequential(
            *_down(3, width, stride=1), *_down(width, width, stride=2), *_down(width, 2 * width, stride=2), *_down(2 * width, 4 * width, stride=2)
        )
        self.decoder = nn.Sequential(
            *_up(4 * width, 2 * width), *_up(2 * width, width), *_up(width, width), nn.Conv2D(width, 3, 3, padding=1), nn.Sigmoid()
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.decoder(self.encoder(x))

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        state = self.state_dict()
        meta = {'image_size': self.image_size, 'width': self.width, **(metadata or {})}
        with SnapshotWriter(path, meta) as snap:
            for name, tensor in state.items():
                snap.declare(name, tensor.shape, tensor.dtype)
            for name, tensor in state.items():
                snap.write(name, tensor)

    @classmethod
    def load(cls, path: str | Path) -> tuple[DenoisingAE, dict[str, Any]]:
        tensors, metadata = deserialize(path)
        model = cls(image_size=int(metadata['image_size']), width=int(metadata['width']))
        device = context.get_default_device()
        state = {name: tensor if device == 'cpu' else tensor.transfer(device) for name, tensor in tensors.items()}
        model.load_state_dict(state)
        return model, metadata
