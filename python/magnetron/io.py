# +---------------------------------------------------------------------+
# | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from pathlib import Path
from types import TracebackType

from ._context import active_context
from ._tensor import Tensor
from ._bootstrap import FFI, C

class StorageArchive:
    def __init__(self, path: str | Path, mode: str) -> None:
        assert str(path).endswith('.mag')
        assert mode in ('r', 'w')
        self._ctx = active_context()
        self._path = str(path)
        self._mode = mode
        self._ptr = FFI.NULL

    def __enter__(self) -> StorageArchive:
        self._ptr = C.mag_storage_archive_open(self._ctx.native_ptr, self._path.encode('utf-8'), bytes(self._mode, 'utf-8'))
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        if self._ptr != FFI.NULL:
            C.mag_storage_archive_close(self._ptr)
            self._ptr = FFI.NULL

    def has_tensor(self, key: str) -> bool:
        return C.mag_storage_archive_has_tensor(self._ptr, key.encode('utf-8'))

    def put_tensor(self, key: str, tensor: Tensor) -> None:
        tensor = tensor.contiguous()
        if not C.mag_storage_archive_put_tensor(self._ptr, key.encode('utf-8'), tensor.native_ptr):
            raise RuntimeError(f'Failed to write tensor with key "{key}" into storage archive at {self._path}')

    def get_tensor(self, key: str) -> Tensor | None:
        ptr = C.mag_storage_archive_get_tensor(self._ptr, key.encode('utf-8'))
        if ptr == FFI.NULL:
            return None
        return Tensor(ptr)

    def write(self, path: str | Path) -> None:
        path = Path(path)
        if not C.mag_storage_archive_write(self._ptr, str(path).encode('utf-8')):
            raise RuntimeError(f'Failed to write storage archive to {path}')
