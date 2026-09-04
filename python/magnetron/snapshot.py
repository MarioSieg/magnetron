# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import datetime
import json
from pathlib import Path
from typing import Any

from ._magnetron_bindings import Tensor, SnapshotStreamWriter as _SnapshotStreamWriter

__all__ = ['serialize', 'deserialize']


def serialize(file_path: Path, tensors: dict[str, Tensor] | Tensor, metadata: dict[str, Any] | None = None):
    if isinstance(tensors, Tensor):
        tensors = {'val': tensors}
    if metadata is None:
        metadata = {}
    offset: int = 0
    def get_offs(t: Tensor) -> int:
        nonlocal offset
        o = offset
        offset += t.numbytes
        return o
    full_metadata = {
        'timestamp': datetime.datetime.now().astimezone().isoformat(),
        'meta': metadata or {},
        'tensormap': {
            key: {
                'shape': tensor.shape,
                'dtype': tensor.dtype.name,
                'offset': get_offs(tensor)
            }
            for key, tensor in tensors.items()
        },
    }
    metadata_json = json.dumps(full_metadata, separators=(',', ':'), ensure_ascii=False)
    nb: int = sum(tensor.numbytes for tensor in tensors.values())
    with _SnapshotStreamWriter(str(file_path), metadata_json, nb) as writer:
        for key, tensor in tensors.items():
            writer.write_tensor(tensor.contiguous().transfer('cpu'))


def deserialize():
    pass
