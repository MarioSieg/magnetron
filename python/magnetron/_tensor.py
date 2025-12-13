# +---------------------------------------------------------------------+
# | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import threading
import weakref
from typing import Any
from collections.abc import Sequence, Callable

from . import context
from .dtype import *
from ._bootstrap import _FFI, _C
from ._error import _handle_errc

_MAIN_TID: int = threading.get_native_id()
_MAX_DIMS: int = 16
_DIM_MAX: int = 0x7FFFFFFFFFFFFFFF

NestedList = float | bool | int | list['NestedData']


def _default_dtype() -> DataType:
    return float32


def _wrap_out_alloc(callback: Callable[[Any], int]) -> _FFI.CData:
    instance: _FFI.CData = _FFI.new(f'mag_tensor_t*[1]')
    status: int = callback(instance)
    _handle_errc(status)
    return instance[0]


def _deduce_tensor_dtype(obj: bool | float | int) -> DataType:
    if isinstance(obj, bool):
        return boolean
    elif isinstance(obj, int):
        return int64
    elif isinstance(obj, float):
        return float32
    else:
        raise TypeError(f'Invalid data type: {type(obj)}')


def _flatten_nested_lists(nested: list[Any]) -> tuple[tuple[int], list[Any]]:
    flat, dims = [], []

    def walk(node: list[Any], depth: int = 0) -> None:
        if isinstance(node, list):
            if len(dims) <= depth:
                dims.append(len(node))
            elif dims[depth] is None or dims[depth] != len(node):
                raise ValueError('All sub-lists must have the same shape')
            for child in node:
                walk(child, depth + 1)
        else:
            if len(dims) <= depth:
                dims.append(None)
            elif dims[depth] is not None:
                raise ValueError('All sub-lists must have the same shape')
            flat.append(node)

    walk(nested)
    return tuple(d for d in dims if d is not None), flat


def _row_major_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    strides = [1] * len(shape)
    for d in range(len(shape) - 2, -1, -1):
        strides[d] = strides[d + 1] * shape[d + 1]
    return tuple(strides)


def _ravel_nested_lists(flat: list[Any], shape: tuple[int], strides: tuple[int], offset: int, dim: int) -> list[Any, ...]:
    if dim == len(shape):
        return flat[offset]
    size = shape[dim]
    stride = strides[dim]
    return [_ravel_nested_lists(flat, shape, strides, offset + i * stride, dim + 1) for i in range(size)]


def _unpack_shape(*dims: int | tuple[int, ...]) -> tuple[int, ...]:
    out: list[int] = []
    def _flatten(obj: int | tuple[int, ...] | list[int]):
        if isinstance(obj, (tuple, list)):
            for y in obj:
                _flatten(y)
        else:
            out.append(int(obj))
    for d in dims:
        _flatten(d)
    return tuple(out)


def _get_reduction_axes(dim: int | Sequence[int] | None) -> tuple[_FFI.CData, int]:
    if dim is None:
        return _FFI.NULL, 0
    if isinstance(dim, int):
        arr = _FFI.new('int64_t[1]', [dim])
        return arr, 1
    if isinstance(dim, Sequence) and not isinstance(dim, str | bytes):
        vals = [int(d) for d in dim]
        if len(vals) == 0:
            dummy = _FFI.new('int64_t[1]', [0])
            return dummy, 0
        arr = _FFI.new(f'int64_t[{len(vals)}]', vals)
        return arr, len(vals)

    raise TypeError('Dimension must be an int, a sequence of ints, or None.')


_SAMPLE_RANGE_DICT: dict[DataType, int | float] = {
    float32: (0.0, 1.0),
    float16: (0.0, 1.0),
    uint8: (0, 2**8 - 1),
    int8: (-(2**7), 2**7 - 1),
    uint16: (0, 2**16 - 1),
    int16: (-(2**15), 2**15 - 1),
    uint32: (0, 2**32 - 1),
    int32: (-(2**31), 2**31 - 1),
    uint64: (0, 2**64 - 1),
    int64: (-(2**63), 2**63 - 1),
}


def _get_uniform_sample_range(dtype: DataType, low: float | int | None = None, high: float | int | None = None) -> tuple[int | float, int | float]:
    if low is None:
        low = _SAMPLE_RANGE_DICT[dtype][0]
    if high is None:
        high = _SAMPLE_RANGE_DICT[dtype][1]
    assert high > low, f'Invalid uniform sample range {high} must be > {low}'
    return low, high


# Variants for indexing into Tensors.
Index = int | slice | type(Ellipsis) | None | object


def _expand_ellipsis(idxs: tuple[Index, ...], rank: int) -> tuple[Index, ...]:
    consuming = sum(1 for x in idxs if x is not None and x is not Ellipsis)
    ellipsis_occurrences = sum(1 for x in idxs if x is Ellipsis)
    if ellipsis_occurrences > 1:
        raise IndexError('Only one Ellipsis (...) is allowed in the index tuple')
    if any(x is Ellipsis for x in idxs):
        ellipsis_pos = next(i for i, x in enumerate(idxs) if x is Ellipsis)
        to_insert = rank - consuming
        if to_insert < 0:
            raise IndexError(f'Too many indices for a tensor of rank {rank}')
        expanded = idxs[:ellipsis_pos] + (slice(None),) * to_insert + idxs[ellipsis_pos + 1 :]
    else:
        if consuming > rank:
            raise IndexError(f'Too many indices for a tensor of rank {rank}')
        if consuming < rank:
            expanded = idxs + (slice(None),) * (rank - consuming)
        else:
            expanded = idxs
    return expanded


class Tensor:
    """A 1-6 dimensional tensor with support for automatic differentiation."""

    __slots__ = ('__weakref__', '_ctx', '_ptr', '_finalizer')

    @property
    def rank(self) -> int:
        return _C.mag_tensor_rank(self._ptr)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(_FFI.unpack(_C.mag_tensor_shape_ptr(self._ptr), self.rank))

    @property
    def strides(self) -> tuple[int, ...]:
        return tuple(_FFI.unpack(_C.mag_tensor_strides_ptr(self._ptr), self.rank))

    @property
    def dtype(self) -> DataType:
        dtype_value: int = _C.mag_tensor_type(self._ptr)
        assert dtype_value in DTYPE_ENUM_MAP, f'Unsupported tensor dtype: {dtype_value}'
        return DTYPE_ENUM_MAP[dtype_value]

    @property
    def data_ptr(self) -> int:
        return int(_FFI.cast('uintptr_t', _C.mag_tensor_data_ptr(self._ptr)))

    @property
    def storage_base_ptr(self) -> int:
        return int(_FFI.cast('uintptr_t', _C.mag_tensor_data_storage_ptr(self._ptr)))

    def item(self) -> float | int | bool:
        if self.numel != 1:
            raise ValueError('Tensor must have exactly one element to retrieve an item')
        scalar_buf = _FFI.new("mag_scalar_t[1]")
        status = _C.mag_tensor_item(self._ptr, scalar_buf)
        if status != _C.MAG_STATUS_OK:
            raise RuntimeError(f"mag_tensor_item failed with status {int(status)}")
        s = scalar_buf[0]
        if _C.mag_scalar_is_f64(s):
            return float(_C.mag_scalar_as_f64(s))
        if _C.mag_scalar_is_i64(s):
            return int(_C.mag_scalar_as_i64(s))
        if _C.mag_scalar_is_u64(s):
            v = _C.mag_scalar_as_u64(s)
            if self.dtype == boolean:
                return bool(v)
            return int(v)
        raise TypeError(f'Unsupported scalar type for item retrieval (scalar.type={int(s.type)})')

    def __bool__(self) -> bool:
        if self.numel != 1:
            raise ValueError('The truth value of a Tensor with more than one element is ambiguous. Use .Any() or .all() instead.')
        return bool(self.item())

    def tolist(self) -> NestedList:
        if self.numel == 0:
            return []
        dt = self.dtype
        native = None
        if dt.is_floating_point:
            tensor = self.cast(float32)
            native = 'float'
        elif dt.is_signed_integer:
            tensor = self.cast(int64)
            native = 'int64_t'
        elif dt.is_unsigned_integer:
            tensor = self.cast(uint64)
            native = 'uint64_t'
        elif dt == boolean:
            tensor = self.cast(boolean)
            native = 'uint8_t'
        else:
            raise TypeError(f'Tensor dtype {self.dtype} is not supported for tolist()')
        ptr = _FFI.cast(f'const {native}*', _C.mag_tensor_copy_data(tensor.native_ptr))
        flat = list(_FFI.unpack(ptr, self.numel))
        _C.mag_tensor_copy_data_free(ptr)
        cont_strides = _row_major_strides(self.shape)
        return _ravel_nested_lists(flat, self.shape, cont_strides, offset=0, dim=0)

    @property
    def data_size(self) -> int:
        return _C.mag_tensor_numbytes(self._ptr)

    @property
    def numel(self) -> int:
        return _C.mag_tensor_numel(self._ptr)

    @property
    def is_transposed(self) -> bool:
        return _C.mag_tensor_is_transposed(self._ptr)

    @property
    def is_permuted(self) -> bool:
        return _C.mag_tensor_is_permuted(self._ptr)

    def is_shape_eq(self, rhs: Tensor) -> bool:
        return _C.mag_tensor_is_shape_eq(self._ptr, rhs._ptr)

    def are_strides_eq(self, rhs: Tensor) -> bool:
        return _C.mag_tensor_are_strides_eq(self._ptr, rhs._ptr)

    def can_broadcast(self, rhs: Tensor) -> bool:
        return _C.mag_tensor_can_broadcast(self._ptr, rhs._ptr)

    @property
    def is_view(self) -> bool:
        return _C.mag_tensor_is_view(self._ptr)

    @property
    def width(self) -> int:
        return self.shape[2]

    @property
    def height(self) -> int:
        return self.shape[1]

    @property
    def channels(self) -> int:
        return self.shape[0]

    @property
    def native_ptr(self) -> _FFI.CData:
        return self._ptr

    @property
    def is_contiguous(self) -> bool:
        return _C.mag_tensor_is_contiguous(self._ptr)

    @property
    def requires_grad(self) -> bool:
        return _C.mag_tensor_requires_grad(self._ptr)

    @requires_grad.setter
    def requires_grad(self, requires: bool) -> None:
        _handle_errc(_C.mag_tensor_set_requires_grad(self._ptr, requires))

    @property
    def grad(self) -> Tensor | None:
        if not self.requires_grad:
            return None
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_tensor_grad(self._ptr, out)))

    def backward(self) -> None:
        _handle_errc(_C.mag_tensor_backward(self._ptr))

    def zero_grad(self) -> None:
        _C.mag_tensor_zero_grad(self._ptr)

    def visualize_backprop_graph(self, file_name: str = 'graph.dot') -> None:
        file_name = bytes(file_name, 'utf-8')
        _C.mag_tensor_visualize_backprop_graph(self._ptr, file_name)

    def __len__(self) -> int:
        return self.shape[0]

    def __str__(self) -> str:
        head, tail = 3, 3
        threshold = 1000 # todo make tose configureable
        cstr: _FFI.CData = _C.mag_tensor_to_string(self._ptr, head, tail, threshold)
        data_str: str = _FFI.string(cstr).decode('utf-8')
        _C.mag_tensor_to_string_free_data(cstr)
        return data_str

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, index: Index | tuple[Index, ...]) -> Tensor:
        if not isinstance(index, tuple):
            index = (index,)
        index = _expand_ellipsis(index, self.rank)
        curr: Tensor = self
        axis: int = 0
        for idx in index:
            if idx is None:
                if curr.rank == _MAX_DIMS:
                    raise NotImplementedError(f'Rank > {_MAX_DIMS} not supported')
                curr = curr.view(curr.shape[:axis], 1, *curr.shape[axis:])
                axis += 1
                continue
            elif isinstance(idx, int):
                dim_size: int = curr.shape[axis]
                if idx < 0:
                    idx += dim_size
                if idx < 0 or idx >= dim_size:
                    raise IndexError(f'Index {idx} is out of bounds for axis {axis} with size {dim_size}')
                curr = curr.view_slice(axis, idx, 1, 1)
                new_shape = list(curr.shape)
                del new_shape[axis]
                if not new_shape:
                    new_shape = [1]
                curr = curr.view(new_shape) if new_shape else curr.view()
                continue
            elif isinstance(idx, slice):
                start, stop, step = idx.indices(curr.shape[axis])
                if step <= 0:
                    raise NotImplementedError('Non-positive slice steps are not supported')
                length = len(range(start, stop, step))
                if length == 0:
                    raise NotImplementedError('Zero-length slice not implemented')
                curr = curr.view_slice(axis, start, length, step)
                axis += 1
                continue
            elif isinstance(idx, Sequence) and not isinstance(idx, Tensor):
                idx = Tensor.of(list(idx), dtype=int64)
            if isinstance(idx, Tensor):
                curr = curr.gather(axis, idx)
                axis += 1
                continue
            raise RuntimeError(f'Invalid index component {idx!r}')
        return curr

    def _validate_inplace_op(self) -> None:
        if context.is_grad_recording() and self.requires_grad:
            raise RuntimeError(
                'In-place operations are not allowed when gradient recording is enabled. '
                'Either disable gradient recording or use the `detach()` method to create a new tensor without gradient tracking.'
            )

    def _expand_rhs(self, rhs: Tensor | int | float | bool) -> Tensor:
        return rhs if isinstance(rhs, Tensor) else Tensor.full_like(self, rhs)

    def _expand_rhs_list(self, rhs: Tensor | int | float | bool | list[int | float | bool]) -> Tensor:
        return Tensor.of(rhs, dtype=self.dtype) if isinstance(rhs, list) else self._expand_rhs(rhs)

    @staticmethod
    def _validate_dtypes(*args: Tensor, allowed_types: set[DataType]) -> None:
        for i, tensor in enumerate(args):
            if not tensor.dtype in allowed_types:
                raise RuntimeError(f'Operation requires dtype {allowed_types} for arg {i + 1} but got {tensor.dtype}')

    def __init__(self, native_object: _FFI.CData | None) -> None:
        assert _MAIN_TID == threading.get_native_id(), 'Context must be created in the main thread'
        self._ctx = context.native_ptr()
        self._ptr = native_object
        self._finalizer = weakref.finalize(self, _C.mag_tensor_decref, self._ptr)

    @classmethod
    def empty(cls, *shape: int | tuple[int, ...], dtype: DataType = _default_dtype(), requires_grad: bool = False) -> Tensor:
        shape: tuple[int, ...] = _unpack_shape(*shape)
        assert 0 <= len(shape) <= _MAX_DIMS, f'Invalid number of dimensions: {len(shape)}'
        assert all(0 < dim <= _DIM_MAX for dim in shape), 'Invalid dimension size'
        dims: _FFI.CData = _FFI.new(f'int64_t[{len(shape)}]', shape)
        instance = _wrap_out_alloc(lambda out: _C.mag_empty(out, context.native_ptr(), dtype.enum_value, len(shape), dims))
        tensor: Tensor = cls(instance)
        tensor.requires_grad = requires_grad
        return tensor

    @classmethod
    def scalar(
        cls, value: int | float | bool, *, dtype: DataType | None = None, requires_grad: bool = False
    ) -> Tensor:
        instance = _wrap_out_alloc(lambda out: _C.mag_scalar(out, context.native_ptr(), dtype.enum_value, _C.mag_scalar_float(value) if isinstance(value, float) else _C.mag_scalar_int(value)))
        tensor: Tensor = cls(instance)
        tensor.requires_grad = requires_grad
        tensor.fill_(value)
        return tensor

    @classmethod
    def empty_like(cls, template: Tensor, *, dtype: DataType | None = None, requires_grad: bool = False) -> Tensor:
        return cls.empty(template.shape, dtype=dtype if dtype is not None else template.dtype, requires_grad=requires_grad)

    @classmethod
    def full(
        cls, *shape: int | tuple[int, ...], fill_value: int | float | bool, dtype: DataType = _default_dtype(), requires_grad: bool = False
    ) -> Tensor:
        shape: tuple[int, ...] = _unpack_shape(*shape)
        tensor: Tensor = cls.empty(
            *shape,
            dtype=dtype,
            requires_grad=requires_grad,
        )
        tensor.fill_(fill_value)
        return tensor

    @classmethod
    def full_like(cls, template: Tensor, fill_value: int | float | bool, *, dtype: DataType | None = None, requires_grad: bool = False) -> Tensor:
        return cls.full(
            template.shape,
            fill_value=fill_value,
            dtype=dtype if dtype is not None else template.dtype,
            requires_grad=requires_grad,
        )

    @classmethod
    def of(cls, data: NestedList, *, dtype: DataType | None = None, requires_grad: bool = False) -> Tensor:
        if isinstance(data, (int, float, bool)):
            if dtype is None:
                dtype = _deduce_tensor_dtype(data)
            return cls.scalar(value=data, dtype=dtype, requires_grad=requires_grad)
        if isinstance(data, (list, tuple)) and len(data) == 0:
            raise ValueError("Tensor.of() does not support empty lists; use Tensor.empty(shape, ...) instead")
        shape, flattened_data = _flatten_nested_lists(data)
        dtype: DataType = dtype if dtype is not None else _deduce_tensor_dtype(flattened_data[0])
        native_name = None
        wide_dtype = None
        if dtype.is_floating_point:
            native_name = 'float'
            wide_dtype = float32
        elif dtype.is_signed_integer:
            native_name = 'int64_t'
            wide_dtype = int64
        elif dtype.is_unsigned_integer:
            native_name = 'uint64_t'
            wide_dtype = uint64
        elif dtype == boolean:
            native_name = 'uint8_t'
            wide_dtype = boolean
        else:
            raise TypeError(f'Tensor dtype {dtype} is not supported for Tensor.of()')
        assert native_name is not None and wide_dtype is not None
        raw: Tensor = cls.empty(*shape, dtype=wide_dtype, requires_grad=requires_grad)
        staging_buffer: _FFI.CData = _FFI.new(f'{native_name}[{len(flattened_data)}]', flattened_data)
        nb = len(flattened_data)*raw.dtype.size
        _handle_errc(_C.mag_copy_raw_(raw.native_ptr, staging_buffer, nb))
        return raw.cast(dtype)

    @classmethod
    def zeros(
        cls,
        *shape: int | tuple[int, ...],
        dtype: DataType = _default_dtype(),
        requires_grad: bool = False,
    ) -> Tensor:
        return cls.full(*shape, fill_value=0, dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def zeros_like(cls, template: Tensor, dtype: DataType | None = None, *, requires_grad: bool = False) -> Tensor:
        return cls.zeros(template.shape, dtype=dtype if dtype is not None else template.dtype, requires_grad=requires_grad)

    @classmethod
    def ones(cls, *shape: int | tuple[int, ...], dtype: DataType = _default_dtype(), requires_grad: bool = False) -> Tensor:
        return cls.full(*shape, fill_value=1, dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def ones_like(cls, template: Tensor, *, dtype: DataType | None = None, requires_grad: bool = False) -> Tensor:
        return cls.ones(template.shape, dtype=dtype if dtype is not None else template.dtype, requires_grad=requires_grad)

    @classmethod
    def uniform(
        cls,
        *shape: int | tuple[int, ...],
        low: float | int | None = None,
        high: float | int | None = None,
        dtype: DataType = _default_dtype(),
        requires_grad: bool = False,
    ) -> Tensor:
        tensor: Tensor = cls.empty(*_unpack_shape(*shape), dtype=dtype, requires_grad=requires_grad)
        tensor.uniform_(low, high)
        return tensor

    @classmethod
    def uniform_like(
        cls,
        template: Tensor,
        *,
        low: float | int | None = None,
        high: float | int | None = None,
        dtype: DataType | None = None,
        requires_grad: bool = False,
    ) -> Tensor:
        return cls.uniform(template.shape, low=low, high=high, dtype=dtype if dtype is not None else template.dtype, requires_grad=requires_grad)

    @classmethod
    def normal(
        cls,
        *shape: int | tuple[int, ...],
        mean: float = 0.0,
        std: float = 1.0,
        dtype: DataType = _default_dtype(),
        requires_grad: bool = False,
    ) -> Tensor:
        tensor: Tensor = cls.empty(*_unpack_shape(*shape), dtype=dtype, requires_grad=requires_grad)
        tensor.normal_(mean, std)
        return tensor

    @classmethod
    def normal_like(
        cls, template: Tensor, *, mean: float = 0.0, std: float = 1.0, dtype: DataType | None = None, requires_grad: bool = False
    ) -> Tensor:
        return cls.normal(template.shape, mean=mean, std=std, dtype=dtype if dtype is not None else template.dtype, requires_grad=requires_grad)

    @classmethod
    def bernoulli(cls, *shape: int | tuple[int, ...], p: float = 0.5) -> Tensor:
        tensor: Tensor = cls.empty(*_unpack_shape(*shape), dtype=boolean, requires_grad=False)
        tensor.bernoulli_(p)
        return tensor

    @classmethod
    def bernoulli_like(cls, template: Tensor, *, p: float = 0.5) -> Tensor:
        return cls.bernoulli(template.shape, p=p)

    @classmethod
    def arange(
        cls,
        start: float | int = 0,
        stop: float | int | None = None,
        step: float | int = 1,
        dtype: DataType | None = None,
        requires_grad: bool = False,
    ) -> Tensor:
        if stop is None:
            stop = start
            start = 0
        if dtype is None:
            dtype = _deduce_tensor_dtype(start)

        assert type(start) == type(stop) == type(step), 'start, stop, and step must be of the same type'
        start = _C.mag_scalar_int(start) if isinstance(start, int) else _C.mag_scalar_float(start)
        stop = _C.mag_scalar_int(stop) if isinstance(stop, int) else _C.mag_scalar_float(stop)
        step = _C.mag_scalar_int(step) if isinstance(step, int) else _C.mag_scalar_float(step)
        instance = _wrap_out_alloc(
            lambda out: _C.mag_arange(out, context.native_ptr(), dtype.enum_value, start, stop, step)
        )
        tensor: Tensor = cls(instance)
        tensor.requires_grad = requires_grad
        return tensor

    @classmethod
    def rand_perm(
        cls,
        n: int,
        dtype: DataType = int64,
        requires_grad: bool = False,
    ) -> Tensor:
        instance = _wrap_out_alloc(lambda out: _C.mag_rand_perm(out, context.native_ptr(), dtype.enum_value, n))
        tensor: Tensor = cls(instance)
        tensor.requires_grad = requires_grad
        return tensor

    @classmethod
    def cat(cls, tensors: Sequence[Tensor], dim: int = 0) -> Tensor:
        num_tensors = len(tensors)
        assert num_tensors > 0, 'At least one tensor is required for concatenation'
        ref = tensors[0]
        rank = ref.rank
        assert -rank <= dim < rank, f'Dimension {dim} out of range for rank {rank}'
        dim = dim % rank
        for t in tensors:
            assert t.rank == rank, 'All tensors must have the same rank for concatenation'
            assert t.dtype == ref.dtype
            for d in range(rank):
                if d == dim:
                    continue
                assert t.shape[d] == ref.shape[d], f'All tensors must match on dim {d}: {t.shape[d]} vs {ref.shape[d]}'
        contig = [t.contiguous() for t in tensors]
        tensor_ptrs = _FFI.new(f"mag_tensor_t*[{len(contig)}]")
        for i,t in enumerate(contig):
            tensor_ptrs[i] = t.native_ptr
        out = Tensor(_wrap_out_alloc(lambda out: _C.mag_cat(out, tensor_ptrs, len(contig), dim)))
        del contig
        return out

    @classmethod
    def load_image(cls, path: str, channels: str = 'RGB', resize_to: tuple[int, int] = (0, 0)) -> Tensor:
        assert channels in ('R', 'RG', 'RGB', 'RGBA'), f'Invalid channels specification: {channels}'
        instance = _wrap_out_alloc(
            lambda out: _C.mag_load_image(
                out, context.native_ptr(), bytes(path, 'utf-8'), bytes(channels.upper(), 'utf-8'), resize_to[0], resize_to[1]
            )
        )
        return cls(instance)

    def clone(self) -> Tensor:
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_clone(out, self._ptr)))

    def cast(self, dst_type: DataType) -> Tensor:
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_cast(out, self._ptr, dst_type.enum_value)))

    def view(self, *dims: int | tuple[int, ...]) -> Tensor:
        dims = _unpack_shape(*dims)
        num_dims: int = len(dims)
        view_dims: _FFI.CData = _FFI.new(f'int64_t[{num_dims}]', dims)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_view(out, self._ptr, view_dims, num_dims)))

    def view_slice(self, dim: int, start: int, length: int, step: int = 1) -> Tensor:
        assert 0 <= dim < self.rank, f'Dimension {dim} out of range for tensor with rank {self.rank}'
        assert start >= 0 and length > 0
        assert start + (length - 1) * step < self.shape[dim], (
            f'Slice out of bounds: start={start}, length={length}, step={step}, shape={self.shape[dim]}'
        )
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_view_slice(out, self._ptr, dim, start, length, step)))

    def split(self, split_size: int, dim: int = 0) -> tuple[Tensor, ...]:
        if self.rank == 0:
            raise RuntimeError("split() is not defined for 0-dim tensors")
        if split_size <= 0:
            raise ValueError(f"split_size must be > 0, got {split_size}")
        if dim < 0:
            dim += self.rank
        if dim < 0 or dim >= self.rank:
            raise IndexError(f"split(): dim {dim} out of range for rank {self.rank}")
        size = self.shape[dim]
        if size == 0:
            return ()
        n_chunks = (size + split_size - 1) // split_size  # same as C expected_chunks
        outs = _FFI.new(f"mag_tensor_t *[{n_chunks}]")
        _handle_errc(_C.mag_split(outs, n_chunks, self._ptr, split_size, dim))
        return tuple(Tensor(outs[i]) for i in range(n_chunks))

    def gather(self, dim: int, index: Tensor) -> Tensor:
        assert 0 <= dim < self.rank, f'Dimension {dim} out of range for tensor with rank {self.rank}'
        assert index.dtype == int64, f'Index tensor must be of int64 dtype, but is {index.dtype}'
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_gather(out, self._ptr, dim, index._ptr)))

    def reshape(self, *dims: int | tuple[int, ...]) -> Tensor:
        dims = _unpack_shape(dims)
        num_dims: int = len(dims)
        view_dims: _FFI.CData = _FFI.new(f'int64_t[{num_dims}]', dims)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_reshape(out, self._ptr, view_dims, num_dims)))

    def transpose(self, dim0: int = 0, dim1: int = 1) -> Tensor:
        assert dim0 != dim1, f'Transposition axes must be not equal, but {dim0} == {dim1}'
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_transpose(out, self._ptr, dim0, dim1)))

    def one_hot(self, num_classes: int = -1) -> Tensor:
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_one_hot(out, self._ptr, num_classes)))

    @property
    def T(self) -> Tensor:
        nd = self.rank
        if nd < 2:
            return self
        if nd == 2:
            return self.transpose(0, 1)
        return self.permute(*range(nd - 1, -1, -1))

    def detach(self) -> Tensor:
        _C.mag_tensor_detach(self._ptr)
        return self

    def contiguous(self) -> Tensor:
        if self.is_contiguous:
            return self
        return self.clone()

    def permute(self, *dims: int | tuple[int, ...]) -> Tensor:
        dims = _unpack_shape(*dims)
        assert len(dims) == self.rank, f'Invalid number of axes, require {self.rank}, got {len(dims)}'
        if len(dims) != _MAX_DIMS:
            dims = dims + tuple(range(self.rank, _MAX_DIMS))
        assert len(dims) == _MAX_DIMS
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_permute(out, self._ptr, _FFI.new('int64_t[]', dims), _MAX_DIMS)))

    def squeeze(self, dim: int | None = None) -> Tensor:
        if dim is None:
            return Tensor(_wrap_out_alloc(lambda out: _C.mag_squeeze_all(out, self._ptr)))
        else:
            return Tensor(_wrap_out_alloc(lambda out: _C.mag_squeeze_dim(out, self._ptr, dim)))

    def unsqueeze(self, dim: int) -> Tensor:
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_unsqueeze(out, self._ptr, dim)))

    def flatten(self, start_dim: int = 0, end_dim: int = -1) -> Tensor:
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_flatten(out, self._ptr, start_dim, end_dim)))

    def unflatten(self, dim: int, sizes: list[int]) -> Tensor:
        shape_dims: _FFI.CData = _FFI.new(f'int64_t[{len(sizes)}]', sizes)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_unflatten(out, self._ptr, dim, shape_dims, len(sizes))))

    def narrow(self, dim: int, start: int, length: int) -> Tensor:
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_narrow(out, self._ptr, dim, start, length)))

    def movedim(self, source: int, destination: int) -> Tensor:
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_movedim(out, self._ptr, source, destination)))

    def select(self, dim: int, index: int) -> Tensor:
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_select(out, self._ptr, dim, index)))

    def fill_(self, value: float | int | bool) -> None:
        #self._validate_inplace_op()
        if self.dtype.is_floating_point:
            _handle_errc(_C.mag_fill_(self._ptr, _C.mag_scalar_float(value)))
        else:
            _handle_errc(_C.mag_fill_(self._ptr, _C.mag_scalar_int(int(value))))

    def masked_fill_(self, mask: Tensor, value: float | int | bool) -> None:
        assert mask.dtype == boolean, f'Mask tensor must be of boolean dtype, but is {mask.dtype}'
        self._validate_inplace_op()
        if self.dtype.is_floating_point:
            _handle_errc(_C.mag_masked_fill_(self._ptr, mask._ptr, _C.mag_scalar_float(value)))
        else:
            _handle_errc(_C.mag_masked_fill_(self._ptr, mask._ptr, _C.mag_scalar_int(int(value))))

    def masked_fill(self, mask: Tensor, value: float | int | bool) -> Tensor:
        filled = self.clone()
        filled.requires_grad = False  # TODO
        filled.masked_fill_(mask, value)
        return filled

    def uniform_(self, low: float | int | None = None, high: float | int | None = None) -> None:
        self._validate_dtypes(self, allowed_types=NUMERIC_DTYPES)
        self._validate_inplace_op()
        low, high = _get_uniform_sample_range(self.dtype, low, high)
        if self.dtype.is_floating_point:
            _handle_errc(_C.mag_uniform_(self._ptr, _C.mag_scalar_float(low), _C.mag_scalar_float(high)))
        else:
            if low >= 0 and high > 0x7fffffffffffffff:
                _handle_errc(_C.mag_uniform_(self._ptr, _C.mag_scalar_uint(int(low)), _C.mag_scalar_uint(int(high))))
            else:
                _handle_errc(_C.mag_uniform_(self._ptr, _C.mag_scalar_int(int(low)), _C.mag_scalar_int(int(high))))

    def normal_(self, mean: float, std: float) -> None:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        _handle_errc(_C.mag_normal_(self._ptr, _C.mag_scalar_float(mean), _C.mag_scalar_float(std)))

    def bernoulli_(self, p: float) -> None:
        self._validate_dtypes(self, allowed_types={boolean})
        self._validate_inplace_op()
        _handle_errc(_C.mag_bernoulli_(self._ptr, _C.mag_scalar_float(p)))

    def copy_(self, x: Tensor) -> None:
        assert self.rank == x.rank, f'Tensor ranks do not match: {self.rank} != {x.rank}'
        assert self.is_shape_eq(x), f'Tensor shapes do not match: {self.shape} != {x.shape}'
        assert self.is_contiguous and x.is_contiguous, 'Both tensors must be contiguous for copy operation'
        _handle_errc(_C.mag_copy_raw_(self._ptr, _FFI.cast('void*', x.data_ptr), x.data_size))

    def zeros_(self) -> None:
        self.fill_(0)

    def ones_(self) -> None:
        self.fill_(1)

    def mean(self, dim: int | Sequence[int] | None = None, keepdim: bool = False) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        dims, num_dims = _get_reduction_axes(dim)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_mean(out, self._ptr, dims, num_dims, keepdim)))

    def min(self, dim: int | Sequence[int] | None = None, keepdim: bool = False) -> Tensor:
        self._validate_dtypes(self, allowed_types=NUMERIC_DTYPES)
        dims, num_dims = _get_reduction_axes(dim)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_min(out, self._ptr, dims, num_dims, keepdim)))

    def max(self, dim: int | Sequence[int] | None = None, keepdim: bool = False) -> Tensor:
        self._validate_dtypes(self, allowed_types=NUMERIC_DTYPES)
        dims, num_dims = _get_reduction_axes(dim)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_max(out, self._ptr, dims, num_dims, keepdim)))

    def argmin(self, dim: int | tuple[int] | None = None, keepdim: bool = False) -> Tensor:
        dims, num_dims = _get_reduction_axes(dim)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_argmin(out, self._ptr, dims, num_dims, keepdim)))

    def argmax(self, dim: int | tuple[int] | None = None, keepdim: bool = False) -> Tensor:
        dims, num_dims = _get_reduction_axes(dim)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_argmax(out, self._ptr, dims, num_dims, keepdim)))

    def sum(self, dim: int | Sequence[int] | None = None, keepdim: bool = False) -> Tensor:
        self._validate_dtypes(self, allowed_types=NUMERIC_DTYPES)
        dims, num_dims = _get_reduction_axes(dim)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_sum(out, self._ptr, dims, num_dims, keepdim)))

    def prod(self, dim: int | Sequence[int] | None = None, keepdim: bool = False) -> Tensor:
        self._validate_dtypes(self, allowed_types=NUMERIC_DTYPES)
        dims, num_dims = _get_reduction_axes(dim)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_prod(out, self._ptr, dims, num_dims, keepdim)))

    def any(self, dim: int | Sequence[int] | None = None, keepdim: bool = False) -> Tensor:
        dims, num_dims = _get_reduction_axes(dim)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_any(out, self._ptr, dims, num_dims, keepdim)))

    def all(self, dim: int | Sequence[int] | None = None, keepdim: bool = False) -> Tensor:
        dims, num_dims = _get_reduction_axes(dim)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_all(out, self._ptr, dims, num_dims, keepdim)))

    def topk(self, k: int, dim: int = -1, largest: bool = True, sorted: bool = True) -> tuple[Tensor, Tensor]:
        self._validate_dtypes(self, allowed_types=NUMERIC_DTYPES)
        values: _FFI.CData = _FFI.new(f'mag_tensor_t*[1]')
        indices: _FFI.CData = _FFI.new(f'mag_tensor_t*[1]')
        _handle_errc(_C.mag_topk(values, indices, self._ptr, k, dim, largest, sorted))
        return Tensor(values[0]), Tensor(indices[0])

    def abs(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_abs(out, self._ptr)))

    def abs_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_abs_(out, self._ptr)))

    def neg(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_neg(out, self._ptr)))

    def neg_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_neg_(out, self._ptr)))

    def __neg__(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return self.neg()

    def __round__(self, n: int | None = None) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return self.round()

    def log(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_log(out, self._ptr)))

    def log_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_log_(out, self._ptr)))

    def log10(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_log10(out, self._ptr)))

    def log10_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_log10_(out, self._ptr)))

    def log1p(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_log1p(out, self._ptr)))

    def log1p_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_log1p_(out, self._ptr)))

    def log2(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_log2(out, self._ptr)))

    def log2_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_log2_(out, self._ptr)))

    def sqr(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_sqr(out, self._ptr)))

    def sqr_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_sqr_(out, self._ptr)))

    def rcp(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_rcp(out, self._ptr)))

    def rcp_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_rcp_(out, self._ptr)))

    def sqrt(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_sqrt(out, self._ptr)))

    def sqrt_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_sqrt_(out, self._ptr)))

    def rsqrt(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_rsqrt(out, self._ptr)))

    def rsqrt_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_rsqrt_(out, self._ptr)))

    def sin(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_sin(out, self._ptr)))

    def sin_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_sin_(out, self._ptr)))

    def cos(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_cos(out, self._ptr)))

    def cos_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_cos_(out, self._ptr)))

    def tan(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_tan(out, self._ptr)))

    def tan_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_tan_(out, self._ptr)))

    def asin(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_asin(out, self._ptr)))

    def asin_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_asin_(out, self._ptr)))

    def acos(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_acos(out, self._ptr)))

    def acos_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_acos_(out, self._ptr)))

    def atan(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_atan(out, self._ptr)))

    def atan_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_atan_(out, self._ptr)))

    def sinh(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_sinh(out, self._ptr)))

    def sinh_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_sinh_(out, self._ptr)))

    def cosh(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_cosh(out, self._ptr)))

    def cosh_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_cosh_(out, self._ptr)))

    def tanh(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_tanh(out, self._ptr)))

    def tanh_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_tanh_(out, self._ptr)))

    def asinh(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_asinh(out, self._ptr)))

    def asinh_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_asinh_(out, self._ptr)))

    def acosh(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_acosh(out, self._ptr)))

    def acosh_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_acosh_(out, self._ptr)))

    def atanh(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_atanh(out, self._ptr)))

    def atanh_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_atanh_(out, self._ptr)))

    def step(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_step(out, self._ptr)))

    def step_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_step_(out, self._ptr)))

    def erf(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_erf(out, self._ptr)))

    def erf_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_erf_(out, self._ptr)))

    def erfc(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_erfc(out, self._ptr)))

    def erfc_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_erfc_(out, self._ptr)))

    def exp(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_exp(out, self._ptr)))

    def exp_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_exp_(out, self._ptr)))

    def exp2(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_exp2(out, self._ptr)))

    def exp2_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_exp2_(out, self._ptr)))

    def expm1(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_expm1(out, self._ptr)))

    def expm1_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_expm1_(out, self._ptr)))

    def floor(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_floor(out, self._ptr)))

    def floor_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_floor_(out, self._ptr)))

    def ceil(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_ceil(out, self._ptr)))

    def ceil_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_ceil_(out, self._ptr)))

    def round(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_round(out, self._ptr)))

    def round_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_round_(out, self._ptr)))

    def trunc(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_trunc(out, self._ptr)))

    def trunc_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_trunc_(out, self._ptr)))

    def softmax(self, dim: int = -1) -> Tensor:
        if dim != -1:
            raise NotImplementedError('Softmax only supports the last dimension (-1) for now')
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_softmax(out, self._ptr)))

    def softmax_(self, dim: int = -1) -> Tensor:
        if dim != -1:
            raise NotImplementedError('Softmax only supports the last dimension (-1) for now')
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_softmax_(out, self._ptr)))

    def sigmoid(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_sigmoid(out, self._ptr)))

    def sigmoid_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_sigmoid_(out, self._ptr)))

    def hard_sigmoid(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_hard_sigmoid(out, self._ptr)))

    def hard_sigmoid_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_hard_sigmoid(out, self._ptr)))

    def silu(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_silu(out, self._ptr)))

    def silu_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_silu_(out, self._ptr)))

    def relu(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_relu(out, self._ptr)))

    def relu_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_relu_(out, self._ptr)))

    def gelu(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_gelu(out, self._ptr)))

    def gelu_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_gelu_(out, self._ptr)))

    def gelu_approx(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_gelu_approx(out, self._ptr)))

    def gelu_approx_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_gelu_approx_(out, self._ptr)))

    def tril(self, diagonal: int = 0) -> Tensor:
        assert self.rank >= 2, f'Tril requires a rank >= 2 but is {self.rank}'
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_tril(out, self._ptr, diagonal)))

    def tril_(self, diagonal: int = 0) -> Tensor:
        assert self.rank >= 2, f'Tril requires a rank >= 2 but is {self.rank}'
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_tril_(out, self._ptr, diagonal)))

    def triu(self, diagonal: int = 0) -> Tensor:
        assert self.rank >= 2, f'Triu requires a rank >= 2 but is {self.rank}'
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_triu(out, self._ptr, diagonal)))

    def triu_(self, diagonal: int = 0) -> Tensor:
        assert self.rank >= 2, f'Triu requires a rank >= 2 but is {self.rank}'
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_triu_(out, self._ptr, diagonal)))

    def multinomial(self, num_samples: int = 1, replacement: bool = False) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        assert self.rank in (1, 2), f'Multinomial sampling requires a 1D or 2D tensor, but got rank {self.rank}'
        assert num_samples > 0
        assert not replacement, 'Multinomial sampling with replacement is not implemented yet'
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_multinomial(out, self._ptr, num_samples, replacement)))

    def logical_and(self, rhs: Tensor) -> Tensor:
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_and(out, self._ptr, rhs._ptr)))

    def logical_and_(self, rhs: Tensor) -> Tensor:
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_and_(out, self._ptr, rhs._ptr)))

    def logical_or(self, rhs: Tensor) -> Tensor:
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_or(out, self._ptr, rhs._ptr)))

    def logical_or_(self, rhs: Tensor) -> Tensor:
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_or_(out, self._ptr, rhs._ptr)))

    def logical_xor(self, rhs: Tensor) -> Tensor:
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_xor(out, self._ptr, rhs._ptr)))

    def logical_xor_(self, rhs: Tensor) -> Tensor:
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_xor_(out, self._ptr, rhs._ptr)))

    def logical_not(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_not(out, self._ptr)))

    def logical_not_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=INTEGRAL_DTYPES)
        self._validate_inplace_op()
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_not_(out, self._ptr)))

    def bitwise_and(self, rhs: Tensor) -> Tensor:
        return self.logical_and(rhs)

    def bitwise_and_(self, rhs: Tensor) -> Tensor:
        return self.logical_and_(rhs)

    def bitwise_or(self, rhs: Tensor) -> Tensor:
        return self.logical_and(rhs)

    def bitwise_or_(self, rhs: Tensor) -> Tensor:
        return self.logical_and_(rhs)

    def bitwise_xor(self, rhs: Tensor) -> Tensor:
        return self.logical_and(rhs)

    def bitwise_xor_(self, rhs: Tensor) -> Tensor:
        return self.logical_and_(rhs)

    def bitwise_not(self) -> Tensor:
        return self.logical_not()

    def bitwise_not_(self) -> Tensor:
        return self.logical_not_()

    def __add__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_add(out, self._ptr, rhs._ptr)))

    def __radd__(self, rhs: int | float) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return rhs + self

    def __iadd__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_add_(out, self._ptr, rhs._ptr)))

    def __sub__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_sub(out, self._ptr, rhs._ptr)))

    def __rsub__(self, rhs: int | float) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return rhs - self

    def __isub__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_sub_(out, self._ptr, rhs._ptr)))

    def __mul__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_mul(out, self._ptr, rhs._ptr)))

    def __rmul__(self, rhs: int | float) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return rhs * self

    def __imul__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_mul_(out, self._ptr, rhs._ptr)))

    def __truediv__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_div(out, self._ptr, rhs._ptr)))

    def __rtruediv__(self, rhs: int | float) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return rhs / self

    def __itruediv__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_div_(out, self._ptr, rhs._ptr)))

    def __floordiv__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_floordiv(out, self._ptr, rhs._ptr)))

    def __rfloordiv__(self, rhs: int | float) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return rhs // self

    def __ifloordiv__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_floordiv_(out, self._ptr, rhs._ptr)))

    def __mod__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_mod(out, self._ptr, rhs._ptr)))

    def __rmod__(self, rhs: int | float) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return rhs % self

    def __imod__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_mod_(out, self._ptr, rhs._ptr)))

    def __matmul__(self, rhs: Tensor) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_matmul(out, self._ptr, rhs._ptr)))

    def __imatmul__(self, rhs: Tensor) -> Tensor:
        raise NotImplementedError('In-place matrix multiplication is not supported')

    def __and__(self, rhs: Tensor | bool | int) -> Tensor:
        rhs = self._expand_rhs(rhs)
        return self.logical_and(rhs)

    def __rand__(self, rhs: int | bool) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return rhs & self

    def __iand__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        return self.logical_and_(rhs)

    def __or__(self, rhs: Tensor | bool | int) -> Tensor:
        rhs = self._expand_rhs(rhs)
        return self.logical_or(rhs)

    def __ror__(self, rhs: int | bool) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return rhs | self

    def __ior__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return self.logical_or_(rhs)

    def __xor__(self, rhs: Tensor | bool | int) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return self.logical_xor(rhs)

    def __rxor__(self, rhs: int | bool) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return rhs ^ self

    def __ixor__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return self.logical_xor_(rhs)

    def __invert__(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=INTEGRAL_DTYPES)
        return self.logical_not()

    def __lshift__(self, rhs: Tensor | int) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_shl(out, self._ptr, rhs._ptr)))

    def __rlshift__(self, rhs: int) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return rhs << self

    def __ilshift__(self, rhs: Tensor | int) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_shl_(out, self._ptr, rhs._ptr)))

    def __rshift__(self, rhs: Tensor | int) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_shr(out, self._ptr, rhs._ptr)))

    def __rrshift__(self, rhs: int) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return rhs >> self

    def __irshift__(self, rhs: Tensor | int) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_shr_(out, self._ptr, rhs._ptr)))

    def __eq__(self, rhs: Tensor | list[int | float | bool] | int | float | bool) -> Tensor:
        rhs = self._expand_rhs_list(rhs)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_eq(out, self._ptr, rhs._ptr)))

    def __ne__(self, rhs: Tensor | list[int | float | bool] | int | float | bool) -> Tensor:
        rhs = self._expand_rhs_list(rhs)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_ne(out, self._ptr, rhs._ptr)))

    def __le__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_le(out, self._ptr, rhs._ptr)))

    def __ge__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_ge(out, self._ptr, rhs._ptr)))

    def __lt__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_lt(out, self._ptr, rhs._ptr)))

    def __gt__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_wrap_out_alloc(lambda out: _C.mag_gt(out, self._ptr, rhs._ptr)))
