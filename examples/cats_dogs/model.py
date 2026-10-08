# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from pathlib import Path
from typing import Any

from magnetron import Tensor, context, nn
from magnetron.snapshot import SnapshotWriter, deserialize

CLASSES: tuple[str, ...] = ('cat', 'dog')


def preprocess(images: Tensor) -> Tensor:
    return images.cast(context.get_default_dtype()) / 127.5 - 1.0


def _conv_block(in_channels: int, out_channels: int, stride: int) -> list[nn.Module]:
    return [nn.Conv2D(in_channels, out_channels, 3, stride=stride, padding=1, bias=False), nn.GroupNorm(8, out_channels), nn.ReLU()]


class CatsDogsNet(nn.Module):
    def __init__(self, image_size: int = 64, width: int = 32, num_classes: int = len(CLASSES)) -> None:
        super().__init__()
        if image_size % 16 != 0 or width % 8 != 0:
            raise ValueError(f'image_size must be a multiple of 16 and width a multiple of 8, but got {image_size} and {width}')
        self.image_size = image_size
        self.width = width
        self.num_classes = num_classes
        self.features = nn.Sequential(
            *_conv_block(3, width, stride=1),
            *_conv_block(width, width, stride=2),
            *_conv_block(width, 2 * width, stride=1),
            *_conv_block(2 * width, 2 * width, stride=2),
            *_conv_block(2 * width, 4 * width, stride=1),
            *_conv_block(4 * width, 4 * width, stride=2),
            *_conv_block(4 * width, 8 * width, stride=2),
        )
        spatial = image_size // 16
        self.classifier = nn.Sequential(nn.Flatten(), nn.Linear(8 * width * spatial * spatial, 256), nn.ReLU(), nn.Linear(256, num_classes))

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.features(x))

    def save(self, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
        state = self.state_dict()
        meta = {'image_size': self.image_size, 'width': self.width, 'classes': list(CLASSES), **(metadata or {})}
        with SnapshotWriter(path, meta) as snap:
            for name, tensor in state.items():
                snap.declare(name, tensor.shape, tensor.dtype)
            for name, tensor in state.items():
                snap.write(name, tensor)

    @classmethod
    def load(cls, path: str | Path) -> tuple[CatsDogsNet, dict[str, Any]]:
        tensors, metadata = deserialize(path)
        model = cls(image_size=int(metadata['image_size']), width=int(metadata['width']), num_classes=len(metadata['classes']))
        device = context.get_default_device()
        state = {name: tensor if device == 'cpu' else tensor.transfer(device) for name, tensor in tensors.items()}
        model.load_state_dict(state)
        return model, metadata
