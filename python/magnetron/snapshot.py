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
import math
from pathlib import Path
from typing import Any, override, Iterable
from dataclasses import asdict, dataclass
from . import __version__ as _mag_version, __snapshot_version__ as _snap_version
from ._magnetron_bindings import (
    Tensor,
    SnapshotStreamWriter as _SnapshotStreamWriter,
    _SNAPSHOT_TBLOB_ALIGN as _TBLOB_ALIGN,
)

__all__ = ['SnapshotWriter', 'deserialize']

def _align_up(x: int, a: int) -> int:
    return (x + a - 1) & ~(a - 1)

@dataclass
class _TensorMetadata:
    shape: tuple[int, ...]
    dtype: str
    offset: int

# JSON manifest version
_MANIFEST_VERSION: int = 1

@dataclass
class _FileManifest:
    manifest_ver: int
    timestamp: str
    magnetron_ver: str
    snapshot_ver: str
    usr_metadata: dict[str, Any]
    tensor_map: dict[str, _TensorMetadata]

    def serialize(self) -> str:
        return json.dumps(asdict(self), separators=(',', ':'), ensure_ascii=False)

    @classmethod
    def deserialize(cls, data: str) -> _FileManifest:
        obj = json.loads(data)
        return cls(
            manifest_ver=obj['manifest_ver'],
            timestamp=obj['timestamp'],
            magnetron_ver=obj['magnetron_ver'],
            snapshot_ver=obj['snapshot_ver'],
            usr_metadata=obj['usr_metadata'],
            tensor_map={
                k: _TensorMetadata(
                    shape=v['shape'],
                    dtype=v['dtype'],
                    offset=v['offset']
                ) for k, v in obj['tensor_map'].items()
            }
        )

@dataclass(frozen=True, slots=True)
class TensorSpec:
    shape: tuple[int, ...]
    dtype: Any

    @property
    def numel(self) -> int:
        return math.prod(self.shape)

    @property
    def numbytes(self) -> int:
        return self.dtype.size * self.numel

class SnapshotWriter:
    def __init__(
        self,
        file_path: str | Path,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._file_path: Path = Path(file_path)
        self._usr_metadata: dict[str, Any] = dict(metadata or {})
        self._specs: dict[str, TensorSpec] = {}
        self._order: list[str] = []
        self._offsets: dict[str, int] = {}
        self._blob_len: int = 0
        self._needle: int = 0
        self._stream: _SnapshotStreamWriter | None = None
        self._done: bool = False

    @property
    def is_sealed(self) -> bool:
        return self._stream is not None

    def declare(self, name: str, shape: Iterable[int], dtype: Any) -> TensorSpec:
        if self.is_sealed:
            raise RuntimeError(f'Cannot declare Tensor with name {name}, writer is sealed')
        if name in self._specs:
            raise KeyError(f'Duplicate tensor with name {name}')
        spec = TensorSpec(
            shape=tuple(int(dim) for dim in shape),
            dtype=dtype
        )
        if any(dim < 0 for dim in spec.shape):
            raise ValueError(f'Tensor {name} has negaive dim in shape {spec.shape}')
        self._specs[name] = spec
        self._order.append(name)
        return spec

    def _plan(self) -> tuple[dict[str, int], int]:
        offsets: dict[str, int] = {}
        end: int = 0
        for key in self._order:
            end = _align_up(end, _TBLOB_ALIGN)
            offsets[key] = end
            end += self._specs[key].numbytes
        return offsets, end

    def _encode_manifest(self) -> _FileManifest:
        return _FileManifest(
            manifest_ver=_MANIFEST_VERSION,
            timestamp=datetime.datetime.now().astimezone().isoformat(),
            magnetron_ver=_mag_version,
            snapshot_ver=_snap_version,
            usr_metadata=self._usr_metadata,
            tensor_map={
                key: _TensorMetadata(
                    shape=self._specs[key].shape,
                    dtype=self._specs[key].dtype.name,
                    offset=self._offsets[key],
                ) for key in self._order
            }
        )

    @property
    def pending(self) -> list[str]:
        return self._order[self._needle:]

    def seal(self) -> None:
        if self.is_sealed:
            return
        if not self._order:
            raise RuntimeError('No tensors declared')
        self._offsets, self._blob_len = self._plan()
        if self._blob_len == 0:
            raise RuntimeError('Empty data section')
        self._stream = _SnapshotStreamWriter(str(self._file_path), self._encode_manifest().serialize(), self._blob_len)

    def write(self, name: str, payload: Any) -> None:
        self.seal()
        assert self._stream is not None
        if self._needle >= len(self._order):
            raise RuntimeError(f'All {len(self._order)} declared tensors were already written, got "{name}"')
        key = self._order[self._needle]
        if name != key:
            raise RuntimeError(f'Writes must follow the declared order because the data section is append-only, expected "{key}" at index {self._needle} but got "{name}"')
        spec = self._specs[name]
        if callable(payload):
            payload = payload()
        if isinstance(payload, Tensor): # Magnetron Tensor
            blob = payload.contiguous().transfer('cpu')
            if tuple(blob.shape) != spec.shape:
                raise RuntimeError(f'"{name}" was declared {spec.shape} but the payload is {tuple(blob.shape)}')
            if blob.dtype != spec.dtype:
                raise RuntimeError(f'"{name}" was declared {spec.dtype.name} but the payload is {blob.dtype.name}')
            if blob.numbytes != spec.numbytes:
                raise RuntimeError(f'"{name}" reserved {spec.numbytes} bytes but the payload spans {blob.numbytes}')
            self._stream.write_tensor(blob)
            del blob
        else: # Membuf case for numpy arrays or torch tensors
            view = memoryview(payload).cast('B')
            if view.nbytes != spec.numbytes:
                raise RuntimeError(f'"{name}" reserved {spec.numbytes} bytes but the buffer spans {view.nbytes}')
            self._stream.write(view)
            view.release()
        del payload
        self._needle += 1

    def abort(self) -> None:
        if self._stream is not None and not self._done:
            self._stream.abort()
        self._done = True

    def close(self) -> None:
        if self._done:
            return
        if self._stream is None:
            raise RuntimeError('Nothing written')
        missing = self.pending
        if missing:
            self.abort()
            raise RuntimeError(f'{len(missing)} declared tensors were never written, first is "{missing[0]}"')
        self._stream.close()
        self._done = True

    def __enter__(self) -> SnapshotWriter:
        return self

    def __exit__(self, exc_type: type[BaseException] | None, *_: Any) -> bool:
        if exc_type is not None:
            self.abort() # Removes temp file aswell
        else:
            self.close()
        return False

def deserialize(file_path: Path) -> tuple[dict[str, Tensor], str]: # Returns tensors, metadata
    pass
