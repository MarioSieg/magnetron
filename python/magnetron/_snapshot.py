# +---------------------------------------------------------------------+
# | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from ._bootstrap import _FFI, _C
from ._tensor import Tensor
from .context import native_ptr

class Snapshot:
    def __init__(self, *, filename: str, mode: str) -> None:
        assert mode in ('r', 'w'), 'Mode must \'r\' for read or \'w\' for write'
        assert filename.endswith('.mag'), 'Filename must end with .mag extension'
        self._mode: str = mode
        self._filename: str = filename
        self._ptr: _FFI.CData | None = None
        self._closed: bool = True

    @classmethod
    def write(cls, filename: str) -> Snapshot:
        return cls(filename=filename, mode='w')

    @classmethod
    def read(cls, filename: str) -> Snapshot:
        return cls(filename=filename, mode='r')

    def __enter__(self) -> Snapshot:
        if not self._closed:
            return self
        ctx = native_ptr()
        if self._mode == 'w':
            self._ptr = _C.mag_snapshot_new(ctx)
        else:
            self._ptr = _C.mag_snapshot_deserialize(ctx, self._filename.encode('utf-8'))
        if self._ptr is None or self._ptr == _FFI.NULL:
            raise RuntimeError('Failed to create or load snapshot')
        self._closed = False
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if self._closed:
            return
        try:
            if not self._closed and self._ptr is not None and self._mode == 'w':
                if not _C.mag_snapshot_serialize(self._ptr, self._filename.encode('utf-8')):
                    raise RuntimeError('Failed to serialize snapshot to file')
            self._ptr = None
            self._closed = True
        finally:
            self.close()

    def _require_open(self) -> None:
        if self._closed or self._ptr is None or self._ptr == _FFI.NULL:
            raise RuntimeError('Snapshot is closed')

    def close(self) -> None:
        if self._closed:
            return
        _C.mag_snapshot_free(self._ptr)
        self._ptr = None
        self._closed = True

    def put_tensor(self, name: str, tensor: Tensor) -> None:
        self._require_open()
        if self._mode != 'w':
            raise RuntimeError('Snapshot is not opened in write mode')
        if not _C.mag_snapshot_put_tensor(self._ptr, name.encode('utf-8'), tensor.native_ptr):
            raise RuntimeError(f'Failed to put tensor "{name}" into snapshot')

    def get_tensor(self, name: str) -> Tensor:
        self._require_open()
        if self._mode != 'r':
            raise RuntimeError('Snapshot is not opened in read mode')
        tensor_ptr = _C.mag_snapshot_get_tensor(self._ptr, name.encode('utf-8'))
        if tensor_ptr is None or tensor_ptr == _FFI.NULL:
            raise RuntimeError(f'Tensor "{name}" not found in snapshot')
        return Tensor(tensor_ptr)

    def tensor_keys(self) -> list[str]:
        self._require_open()
        n_ptr = _FFI.new("size_t[1]")
        keys_pp = _C.mag_snapshot_get_tensor_keys(self._ptr, n_ptr)
        n = int(n_ptr[0])
        if keys_pp == _FFI.NULL or n == 0:
            return []
        try:
            out: list[str] = []
            for i in range(n):
                cstr = keys_pp[i]
                if cstr == _FFI.NULL:
                    continue
                out.append(_FFI.string(cstr).decode("utf-8"))
            return out
        finally:
            _C.mag_snapshot_free_tensor_keys(keys_pp, n)

    def print_info(self) -> None:
        self._require_open()
        _C.mag_snapshot_print_info(self._ptr)
