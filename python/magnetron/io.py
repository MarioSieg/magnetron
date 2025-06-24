# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

from pathlib import Path
from magnetron import Tensor, Context
from magnetron._bootstrap import load_native_module

_ffi, _C = load_native_module()

from pathlib import Path
from magnetron import Tensor, Context
from magnetron._bootstrap import load_native_module

_ffi, _C = load_native_module()


class StorageStream:
    """Reads and writes Magnetron (.mag) storage files.

    Supports context management to ensure proper cleanup of native resources.
    """

    def __init__(self, handle: _ffi.CData | None = None) -> None:
        self._ctx = Context.get()
        if handle is not None:
            if handle == _ffi.NULL:
                raise ValueError('Received an invalid native handle (NULL).')
            self._ptr = handle
        else:
            self._ptr = _C.mag_storage_stream_new(self._ctx.native_ptr)
        self._closed = False

    def close(self) -> None:
        """Closes the storage stream and frees native resources."""
        if not self._closed:
            _C.mag_storage_stream_close(self._ptr)
            self._ptr = None
            self._closed = True

    def __del__(self) -> None:
        self.close()  # Ensure resources are freed if not explicitly closed

    def __enter__(self) -> 'StorageStream':
        """Enters the runtime context related to this object."""
        if self._closed:
            raise RuntimeError('Cannot enter context with a closed StorageStream.')
        return self

    def __exit__(self, exc_type: any, exc_value: any, traceback: any) -> None:
        """Exits the runtime context and closes the storage stream."""
        self.close()

    @property
    def native_ptr(self) -> _ffi.CData:
        """Returns the native pointer associated with this storage stream."""
        if self._closed:
            raise RuntimeError('Attempting to access a closed StorageStream.')
        return self._ptr

    @classmethod
    def open(cls, file_path: Path | str) -> 'StorageStream':
        """Opens a Magnetron storage file for reading."""
        if isinstance(file_path, str):
            file_path = Path(file_path)
        handle = _C.mag_storage_stream_deserialize(Context.get().native_ptr, str(file_path).encode('utf-8'))
        if handle == _ffi.NULL:
            raise RuntimeError(f'Failed to open storage stream from file: {file_path}')
        return cls(handle)

    def put(self, key: str, tensor: Tensor) -> None:
        """Adds a tensor with a unique key to the storage stream."""
        if self._closed:
            raise RuntimeError('Cannot operate on a closed StorageStream.')
        if tensor.native_ptr is None or tensor.native_ptr == _ffi.NULL:
            raise ValueError('Invalid tensor provided.')
        key_utf8: bytes = key.encode('utf-8')
        if _C.mag_storage_stream_get_tensor(self._ptr, key_utf8) != _ffi.NULL:
            raise RuntimeError(f"Tensor with key '{key}' already exists in the storage stream")
        if not _C.mag_storage_stream_put_tensor(self._ptr, key_utf8, tensor.native_ptr):
            raise RuntimeError('Failed to put tensor into storage stream')

    def get(self, key: str) -> Tensor | None:
        """Retrieves a tensor from the storage stream."""
        if self._closed:
            raise RuntimeError('Cannot operate on a closed StorageStream.')
        handle = _C.mag_storage_stream_get_tensor(self._ptr, key.encode('utf-8'))
        if handle == _ffi.NULL:
            return None
        return Tensor(handle)

    def tensor_keys(self) -> list[str]:
        """Returns a list of all tensor keys in the storage stream."""
        count: _ffi.CData = _ffi.new('size_t[1]')
        keys: _ffi.CData = _C.mag_storage_stream_get_all_tensor_keys(self._ptr, count)
        result: list[str] = []
        print(count[0])
        for i in range(count[0]):
            result.append(_ffi.string(keys[i]).decode('utf-8'))
        _C.mag_storage_stream_get_all_tensor_keys_free_data(keys)  # Free the keys data
        return result

    def __getitem__(self, key: str) -> Tensor:
        """Retrieves a tensor from the storage stream."""
        tensor: Tensor | None = self.get(key)
        if tensor is None:
            raise KeyError(f"Tensor with key '{key}' not found in the storage stream")
        return tensor

    def __setitem__(self, key: str, tensor: Tensor) -> None:
        """Adds a tensor with a unique key to the storage stream."""
        self.put(key, tensor)

    def serialize(self, file_path: Path) -> None:
        """Serializes the storage stream to a file."""
        if self._closed:
            raise RuntimeError('Cannot serialize a closed StorageStream.')
        if not _C.mag_storage_stream_serialize(self._ptr, str(file_path).encode('utf-8')):
            raise RuntimeError(f'Failed to serialize storage stream to file: {file_path}')
