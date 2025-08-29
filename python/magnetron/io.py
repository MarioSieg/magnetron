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
        self._ptr = C.mag_storage_open(self._ctx.native_ptr, self._path.encode('utf-8'), bytes(self._mode, 'utf-8'))
        if self._ptr == FFI.NULL:
            raise RuntimeError(f'Failed to open storage archive at {self._path} with mode "{self._mode}"')
        return self

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        if self._ptr != FFI.NULL:
            if not C.mag_storage_close(self._ptr):
                raise RuntimeError(f'Failed to close storage archive at {self._path}')
            self._ptr = FFI.NULL

    def tensor_keys(self) -> list[str]:
        len = FFI.new('size_t[1]')
        keys_ptr = C.mag_storage_get_all_tensor_keys(self._ptr, len)
        keys = [FFI.string(keys_ptr[i]).decode('utf-8') for i in range(len[0])]
        C.mag_storage_get_all_keys_free(keys_ptr, len[0])
        return keys

    def metadata_keys(self) -> list[str]:
        len = FFI.new('size_t[1]')
        keys_ptr = C.mag_storage_get_all_metadata_keys(self._ptr, len)
        keys = [FFI.string(keys_ptr[i]).decode('utf-8') for i in range(len[0])]
        C.mag_storage_get_all_keys_free(keys_ptr, len[0])
        return keys

    def get_tensor(self, key: str) -> Tensor | None:
        ptr = C.mag_storage_get_tensor(self._ptr, key.encode('utf-8'))
        if ptr == FFI.NULL:
            return None
        return Tensor(ptr)

    def set_tensor(self, key: str, tensor: Tensor) -> bool:
        tensor = tensor.contiguous()
        return C.mag_storage_set_tensor(self._ptr, key.encode('utf-8'), tensor.native_ptr)

    def get_metadata(self, key: str) -> int | float | None:
        type = C.mag_storage_get_metadata_type(self._ptr, key.encode('utf-8'))
        if type == C.MAG_RECORD_TYPE_I64:
            value = FFI.new('int64_t[1]')
            if not C.mag_storage_get_metadata_i64(self._ptr, key.encode('utf-8'), value):
                return None
            return int(value[0])
        elif type == C.MAG_RECORD_TYPE_F64:
            value = FFI.new('double[1]')
            if not C.mag_storage_get_metadata_f64(self._ptr, key.encode('utf-8'), value):
                return None
            return float(value[0])
        else:
            return None

    def set_metadata(self, key: str, metadata: int | float) -> bool:
        if isinstance(metadata, int):
            return C.mag_storage_set_metadata_i64(self._ptr, key.encode('utf-8'), metadata)
        elif isinstance(metadata, float):
            return C.mag_storage_set_metadata_f64(self._ptr, key.encode('utf-8'), metadata)
        else:
            return False

    def __getitem__(self, key: str) -> Tensor | int | float:
        data = self.get_tensor(key)
        if data is None:
            data = self.get_metadata(key)
        if data is None:
            raise KeyError(f'Key "{key}" not found in storage archive at {self._path}')
        return data

    def __setitem__(self, key: str, value: Tensor | int | float) -> None:
        if isinstance(value, Tensor):
            if not self.set_tensor(key, value):
                raise RuntimeError(f'Failed to set tensor for key "{key}" in storage archive at {self._path}')
        elif isinstance(value, (int, float)):
            if not self.set_metadata(key, value):
                raise RuntimeError(f'Failed to set metadata for key "{key}" in storage archive at {self._path}')
        else:
            raise TypeError(f'Value must be a Tensor, int, or float, got {type(value)}')

    def tensors(self) -> dict[str, Tensor]:
        return {k: self.get_tensor(k) for k in self.tensor_keys()}

    def metadata(self) -> dict[str, int | float]:
        return {k: self.get_metadata(k) for k in self.metadata_keys()}
