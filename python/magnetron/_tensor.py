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
from typing import Any, Sequence

from ._context import active_context, default_dtype
from ._core import *
from ._dtype import *
from ._bootstrap import FFI, C

_MAIN_TID: int = threading.get_native_id()

NestedList = float | bool | int | list['NestedData']


def _deduce_tensor_dtype(obj: bool | float | int) -> DataType:
    if isinstance(obj, bool):
        return boolean
    elif isinstance(obj, int):
        return int32
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


def _unpack_shape(*shape: int | tuple[int, ...]) -> tuple[int, ...]:
    if len(shape) == 1 and isinstance(shape[0], tuple):
        shape = shape[0]
    assert len(shape) <= MAX_DIMS, f'Invalid number of dimensions: {len(shape)}, maximum is {MAX_DIMS}'
    return shape


def _get_reduction_axes(dim: int | Sequence[int] | None) -> tuple[FFI.CData, int]:
    if dim is None:
        return FFI.NULL, 0
    if isinstance(dim, int):
        arr = FFI.new('int64_t[1]', [dim])
        return arr, 1
    if isinstance(dim, Sequence) and not isinstance(dim, (str, bytes)):
        vals = [int(d) for d in dim]
        if len(vals) == 0:
            dummy = FFI.new('int64_t[1]', [0])
            return dummy, 0
        arr = FFI.new(f'int64_t[{len(vals)}]', vals)
        return arr, len(vals)

    raise TypeError('Dimension must be an int, a sequence of ints, or None.')


def _get_uniform_sample_range(is_int: bool, low: float | int | None = None, high: float | int | None = None) -> tuple[int | float, int | float]:
    if low is None:
        low = -0x80000000 if is_int else 0.0
    if high is None:
        high = 0x7FFFFFFF if is_int else 1.0
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
        return C.mag_tensor_get_rank(self._ptr)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(FFI.unpack(C.mag_tensor_get_shape(self._ptr), self.rank))

    @property
    def strides(self) -> tuple[int, ...]:
        return tuple(FFI.unpack(C.mag_tensor_get_strides(self._ptr), self.rank))

    @property
    def dtype(self) -> DataType:
        dtype_value: int = C.mag_tensor_get_dtype(self._ptr)
        assert dtype_value in DTYPE_ENUM_MAP, f'Unsupported tensor dtype: {dtype_value}'
        return DTYPE_ENUM_MAP[dtype_value]

    @property
    def data_ptr(self) -> int:
        return int(FFI.cast('uintptr_t', C.mag_tensor_get_data_ptr(self._ptr)))

    @property
    def storage_base_ptr(self) -> int:
        return int(FFI.cast('uintptr_t', C.mag_tensor_get_storage_base_ptr(self._ptr)))

    def item(self) -> float | int | bool:
        if self.numel != 1:
            raise ValueError('Tensor must have exactly one element to retrieve an item')
        if self.dtype.is_floating_point:
            return float(C.mag_tensor_get_item_float(self._ptr))
        elif self.dtype == int32:
            return int(C.mag_tensor_get_item_int(self._ptr))
        elif self.dtype == boolean:
            return bool(C.mag_tensor_get_item_bool(self._ptr))
        else:
            raise TypeError(f'Unsupported tensor dtype for item retrieval: {self.dtype}')

    def __bool__(self) -> bool:
        if self.numel != 1:
            raise ValueError('The truth value of a Tensor with more than one element is ambiguous. Use .Any() or .all() instead.')
        return bool(self.item())

    def tolist(self) -> NestedList:
        if self.numel == 0:
            return []
        is_fp: bool = self.dtype.is_floating_point
        unpack_fn = C.mag_tensor_get_data_as_floats if is_fp else C.mag_tensor_get_raw_data_as_bytes
        free_fn = C.mag_tensor_get_data_as_floats_free if is_fp else C.mag_tensor_get_raw_data_as_bytes_free
        ptr = unpack_fn(self._ptr)
        if not is_fp:
            native: str | None = self.dtype.native_type
            assert native is not None, f'Tensor dtype {self.dtype} does not have a native type'
            ptr = FFI.cast(f'const {native}*', ptr)
        flat = list(FFI.unpack(ptr, self.numel))
        free_fn(ptr)
        cont_strides = _row_major_strides(self.shape)
        return _ravel_nested_lists(flat, self.shape, cont_strides, offset=0, dim=0)

    @property
    def data_size(self) -> int:
        return C.mag_tensor_get_data_size(self._ptr)

    @property
    def numel(self) -> int:
        return C.mag_tensor_get_numel(self._ptr)

    @property
    def is_transposed(self) -> bool:
        return C.mag_tensor_is_transposed(self._ptr)

    @property
    def is_permuted(self) -> bool:
        return C.mag_tensor_is_permuted(self._ptr)

    def is_shape_eq(self, rhs: Tensor) -> bool:
        return C.mag_tensor_is_shape_eq(self._ptr, rhs._ptr)

    def are_strides_eq(self, rhs: Tensor) -> bool:
        return C.mag_tensor_are_strides_eq(self._ptr, rhs._ptr)

    def can_broadcast(self, rhs: Tensor) -> bool:
        return C.mag_tensor_can_broadcast(self._ptr, rhs._ptr)

    @property
    def is_view(self) -> bool:
        return C.mag_tensor_is_view(self._ptr)

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
    def native_ptr(self) -> FFI.CData:
        return self._ptr

    @property
    def is_contiguous(self) -> bool:
        return C.mag_tensor_is_contiguous(self._ptr)

    @property
    def requires_grad(self) -> bool:
        return C.mag_tensor_requires_grad(self._ptr)

    @requires_grad.setter
    def requires_grad(self, require: bool) -> None:
        if require and not self.dtype.is_floating_point:
            raise RuntimeError(f'Tensors requiring gradients must be of a floating point type, but is: {self.dtype}')
        C.mag_tensor_set_requires_grad(self._ptr, require)

    @property
    def grad(self) -> Tensor | None:
        if not self.requires_grad:
            return None
        ptr: FFI.CData = C.mag_tensor_get_grad(self._ptr)
        if ptr is None or ptr == FFI.NULL:
            return None
        return Tensor(ptr)

    def backward(self) -> None:
        assert self.requires_grad, 'Tensor must require gradient tracking'
        assert self.rank == 1 and self.numel == 1, 'Tensor must be scalar'
        C.mag_tensor_backward(self._ptr)

    def zero_grad(self) -> None:
        assert self.requires_grad, 'Tensor must require gradient tracking'
        C.mag_tensor_zero_grad(self._ptr)

    def dump_graph_dot(self, file_path: str, forward: bool) -> None:
        file_path = bytes(file_path, 'utf-8')
        if forward:
            C.mag_tensor_export_forward_graph_graphviz(self._ptr, file_path)
        else:
            C.mag_tensor_export_backward_graph_graphviz(self._ptr, file_path)

    def __len__(self) -> int:
        return self.shape[0]

    def __str__(self) -> str:
        cstr: FFI.CData = C.mag_tensor_to_string(self._ptr, False, 0, 0)
        data_str: str = FFI.string(cstr).decode('utf-8')
        C.mag_tensor_to_string_free_data(cstr)
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
                if curr.rank == MAX_DIMS:
                    raise NotImplementedError('Rank > 6 not supported')
                curr = curr.view(*curr.shape[:axis], 1, *curr.shape[axis:])
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
                curr = curr.view(*new_shape) if new_shape else curr.view()
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
                idx = Tensor.of(list(idx), dtype=int32)

            if isinstance(idx, Tensor):
                if idx.dtype != int32:
                    raise RuntimeError(f'Tensor index must be int32, got {idx.dtype}')
                curr = curr.gather(axis, idx)
                axis += 1
                continue
            raise RuntimeError(f'Invalid index component {idx!r}')
        return curr

    def __setitem__(self, indices: int | tuple[int, ...], value: float) -> None:
        if isinstance(indices, int):
            C.mag_tensor_subscript_set_flattened(self._ptr, indices, float(value))
        elif isinstance(indices, tuple):
            idx = indices + (0,) * (MAX_DIMS - len(indices))
            C.mag_tensor_subscript_set_multi(self._ptr, *idx, float(value))
        else:
            raise TypeError('Indices must be an int or a tuple of ints.')

    def _validate_inplace_op(self) -> None:
        if active_context().is_grad_recording and self.requires_grad:
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

    def __init__(self, native_object: FFI.CData | None) -> None:
        assert _MAIN_TID == threading.get_native_id(), 'Context must be created in the main thread'
        self._ctx = active_context()
        self._ptr = native_object
        self._finalizer = weakref.finalize(self, C.mag_tensor_decref, self._ptr)

    @classmethod
    def empty(cls, *shape: int | tuple[int, ...], dtype: DataType = default_dtype(), requires_grad: bool = False) -> Tensor:
        shape: tuple[int, ...] = _unpack_shape(*shape)
        assert 0 < len(shape) <= MAX_DIMS, f'Invalid number of dimensions: {len(shape)}'
        assert all(0 < dim <= DIM_MAX for dim in shape), 'Invalid dimension size'
        dims: FFI.CData = FFI.new(f'int64_t[{len(shape)}]', shape)
        instance: FFI.CData = C.mag_tensor_empty(active_context().native_ptr, dtype.enum_value, len(shape), dims)
        tensor: Tensor = cls(instance)
        tensor.requires_grad = requires_grad
        return tensor

    @classmethod
    def empty_like(cls, template: Tensor, *, dtype: DataType | None = None, requires_grad: bool = False) -> Tensor:
        return cls.empty(template.shape, dtype=dtype if dtype is not None else template.dtype, requires_grad=requires_grad)

    @classmethod
    def full(
        cls, *shape: int | tuple[int, ...], fill_value: int | float | bool, dtype: DataType = default_dtype(), requires_grad: bool = False
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
        if not data:
            return cls.empty(0, dtype=dtype if dtype is not None else default_dtype())
        shape, flattened_data = _flatten_nested_lists(data)
        dtype: DataType = dtype if dtype is not None else _deduce_tensor_dtype(flattened_data[0])
        native_name: str = dtype.native_type
        alloc_fn: FFI.CData = dtype.fill_fn
        tensor: Tensor = cls.empty(*shape, dtype=dtype, requires_grad=requires_grad)
        staging_buffer: FFI.CData = FFI.new(f'{native_name}[{len(flattened_data)}]', flattened_data)
        copy_bytes_numel: int = len(flattened_data)
        if (
            alloc_fn == C.mag_tensor_fill_from_raw_bytes
        ):  # If the dtype is not a floating point type, we need to multiply by the size of the dtype for the raw bytes initializer.
            copy_bytes_numel *= dtype.size
        alloc_fn(tensor._ptr, staging_buffer, copy_bytes_numel)
        return tensor

    @classmethod
    def zeros(
        cls,
        *shape: int | tuple[int, ...],
        dtype: DataType = default_dtype(),
        requires_grad: bool = False,
    ) -> Tensor:
        return cls.full(*shape, fill_value=0, dtype=dtype, requires_grad=requires_grad)

    @classmethod
    def zeros_like(cls, template: Tensor, dtype: DataType | None = None, *, requires_grad: bool = False) -> Tensor:
        return cls.zeros(template.shape, dtype=dtype if dtype is not None else template.dtype, requires_grad=requires_grad)

    @classmethod
    def ones(cls, *shape: int | tuple[int, ...], dtype: DataType = default_dtype(), requires_grad: bool = False) -> Tensor:
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
        dtype: DataType = default_dtype(),
        requires_grad: bool = False,
    ) -> Tensor:
        tensor: Tensor = cls.empty(*_unpack_shape(*shape), dtype=dtype, requires_grad=requires_grad)
        tensor.fill_random_uniform_(low, high)
        return tensor

    @classmethod
    def normal(
        cls,
        *shape: int | tuple[int, ...],
        mean: float = 0.0,
        std: float = 1.0,
        dtype: DataType = default_dtype(),
        requires_grad: bool = False,
    ) -> Tensor:
        tensor: Tensor = cls.empty(*_unpack_shape(*shape), dtype=dtype, requires_grad=requires_grad)
        tensor.fill_random_normal_(mean, std)
        return tensor

    @classmethod
    def bernoulli(cls, *shape: int | tuple[int, ...], p: float = 0.5) -> Tensor:
        tensor: Tensor = cls.empty(*_unpack_shape(*shape), dtype=boolean, requires_grad=False)
        tensor.fill_random_bernoulli_(p)
        return tensor

    @classmethod
    def arange(
        cls,
        start: float | int = 0,
        stop: float | int | None = None,
        step: float | int = 1,
        dtype: DataType = default_dtype(),
        requires_grad: bool = False,
    ) -> Tensor:
        if stop is None:
            stop = start
            start = 0
        tensor: Tensor = cls.empty((stop - start + step - 1) // step, dtype=dtype, requires_grad=requires_grad)
        tensor.fill_arange_(start, step)
        return tensor

    def clone(self) -> Tensor:
        return Tensor(C.mag_clone(self._ptr))

    def view(self, *dims: int | tuple[int, ...]) -> Tensor:
        dims = _unpack_shape(dims)
        num_dims: int = len(dims)
        view_dims: FFI.CData = FFI.new(f'int64_t[{num_dims}]', dims)
        return Tensor(C.mag_view(self._ptr, view_dims, num_dims))

    def view_slice(self, dim: int, start: int, length: int, step: int = 1) -> Tensor:
        assert 0 <= dim < self.rank, f'Dimension {dim} out of range for tensor with rank {self.rank}'
        assert start >= 0 and length > 0
        assert start + (length - 1) * step < self.shape[dim], (
            f'Slice out of bounds: start={start}, length={length}, step={step}, shape={self.shape[dim]}'
        )
        return Tensor(C.mag_view_slice(self._ptr, dim, start, length, step))

    def split(self, chunk_size: int, dim: int = 0) -> tuple[Tensor, ...]:
        assert chunk_size > 0, 'chunk_size must be greater than 0, got {chunk_size}'
        assert -self.rank <= dim < self.rank, f'Dimension {dim} out of range for tensor with rank {self.rank}'
        dim: int = dim % self.rank
        size: int = self.shape[dim]
        n_chunks: int = (size + chunk_size - 1) // chunk_size
        chunks: list[Tensor] = []
        start: int = 0
        for _ in range(n_chunks):
            length = min(chunk_size, size - start)
            view = self.view_slice(dim, start, length, 1)
            chunks.append(view)
            start += chunk_size
        return tuple(chunks)

    def gather(self, dim: int, index: Tensor) -> Tensor:
        assert 0 <= dim < self.rank, f'Dimension {dim} out of range for tensor with rank {self.rank}'
        assert index.dtype == int32, f'Index tensor must be of int32 dtype, but is {index.dtype}'
        return Tensor(C.mag_gather(self._ptr, dim, index._ptr))

    def reshape(self, *dims: int | tuple[int, ...]) -> Tensor:
        dims = _unpack_shape(dims)
        num_dims: int = len(dims)
        view_dims: FFI.CData = FFI.new(f'int64_t[{num_dims}]', dims)
        return Tensor(C.mag_reshape(self._ptr, view_dims, num_dims))

    def transpose(self, dim1: int = 0, dim2: int = 1) -> Tensor:
        assert dim1 != dim2, f'Transposition axes must be not equal, but {dim1} == {dim2}'
        return Tensor(C.mag_transpose(self._ptr, dim1, dim2))

    @property
    def T(self) -> Tensor:
        return Tensor(C.mag_transpose(self._ptr, 0, 1))

    def detach(self) -> Tensor:
        C.mag_tensor_detach(self._ptr)
        return self

    def contiguous(self) -> Tensor:
        if self.is_contiguous:
            return self
        return self.clone()

    def permute(self, *dims: int | tuple[int, ...]) -> Tensor:
        dims = _unpack_shape(*dims)
        assert len(dims) == self.rank, f'Invalid number of axes, require {self.rank}, got {len(dims)}'
        if len(dims) != MAX_DIMS:
            dims = dims + tuple(range(self.rank, MAX_DIMS))
        assert len(dims) == MAX_DIMS
        dims = FFI.new('int64_t[]', dims)
        for i in range(MAX_DIMS):
            assert 0 <= dims[i] < MAX_DIMS
            for j in range(i + 1, MAX_DIMS):
                assert dims[i] != dims[j], f'Duplicate axis: {dims[i]}'
        return Tensor(C.mag_permute(self._ptr, dims, MAX_DIMS))

    def fill_(self, value: float | int | bool) -> None:
        self._validate_inplace_op()
        if self.dtype.is_floating_point:
            C.mag_tensor_fill_float(self._ptr, float(value))
        else:
            C.mag_tensor_fill_int(self._ptr, int(value))

    def masked_fill_(self, mask: Tensor, value: float | int | bool) -> None:
        assert mask.dtype == boolean, f'Mask tensor must be of boolean dtype, but is {mask.dtype}'
        self._validate_inplace_op()
        if self.dtype.is_floating_point:
            C.mag_tensor_masked_fill_float(self._ptr, mask._ptr, float(value))
        else:
            C.mag_tensor_masked_fill_int(self._ptr, mask._ptr, int(value))

    def masked_fill(self, mask: Tensor, value: float | int | bool) -> Tensor:
        filled = self.clone()
        filled.requires_grad = False  # TODO
        filled.masked_fill_(mask, value)
        return filled

    def fill_random_uniform_(self, low: float | int | None = None, high: float | int | None = None) -> None:
        self._validate_dtypes(self, allowed_types=NUMERIC_DTYPES)
        self._validate_inplace_op()
        low, high = _get_uniform_sample_range(self.dtype.is_integer, low, high)
        if self.dtype.is_floating_point:
            C.mag_tensor_fill_random_uniform_float(self._ptr, low, high)
        else:
            C.mag_tensor_fill_random_uniform_int(self._ptr, low, high)

    def fill_random_normal_(self, mean: float, std: float) -> None:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        C.mag_tensor_fill_random_normal(self._ptr, mean, std)

    def fill_random_bernoulli_(self, p: float) -> None:
        self._validate_dtypes(self, allowed_types={boolean})
        self._validate_inplace_op()
        C.mag_tensor_fill_random_bernoulli(self._ptr, p)

    def fill_arange_(self, start: float | int = 0, step: float | int = 1) -> None:
        self._validate_dtypes(self, allowed_types=NUMERIC_DTYPES)
        self._validate_inplace_op()
        assert self.rank == 1, f'Tensor must be 1-dimensional for arange fill, but is {self.rank}-dimensional'
        C.mag_tensor_fill_arange(self._ptr, float(start), float(step))

    def copy_(self, x: Tensor) -> None:
        assert self.rank == x.rank, f'Tensor ranks do not match: {self.rank} != {x.rank}'
        assert self.is_shape_eq(x), f'Tensor shapes do not match: {self.shape} != {x.shape}'
        assert self.is_contiguous and x.is_contiguous, 'Both tensors must be contiguous for copy operation'
        C.mag_tensor_fill_from_raw_bytes(self._ptr, FFI.cast('void*', x.data_ptr), x.data_size)

    def zeros_(self) -> None:
        self.fill_(0)

    def ones_(self) -> None:
        self.fill_(1)

    def mean(self, dim: int | Sequence[int] | None = None, keepdim: bool = False) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        dims, num_dims = _get_reduction_axes(dim)
        return Tensor(C.mag_mean(self._ptr, dims, num_dims, keepdim))

    def min(self, dim: int | Sequence[int] | None = None, keepdim: bool = False) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        dims, num_dims = _get_reduction_axes(dim)
        return Tensor(C.mag_min(self._ptr, dims, num_dims, keepdim))

    def max(self, dim: int | Sequence[int] | None = None, keepdim: bool = False) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        dims, num_dims = _get_reduction_axes(dim)
        return Tensor(C.mag_max(self._ptr, dims, num_dims, keepdim))

    def sum(self, dim: int | Sequence[int] | None = None, keepdim: bool = False) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        dims, num_dims = _get_reduction_axes(dim)
        return Tensor(C.mag_sum(self._ptr, dims, num_dims, keepdim))

    def argmin(self, dim: int | tuple[int] | None = None, keepdim: bool = False) -> Tensor:
        raise NotImplementedError('argmin is not implemented for complex tensors')
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        dims, num_dims = _get_reduction_axes(dim)
        return Tensor(C.mag_argmin(self._ptr, dims, num_dims, keepdim))

    def argmax(self, dim: int | tuple[int] | None = None, keepdim: bool = False) -> Tensor:
        raise NotImplementedError('argmin is not implemented for complex tensors')
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        dims, num_dims = _get_reduction_axes(dim)
        return Tensor(C.mag_argmax(self._ptr, dims, num_dims, keepdim))

    def abs(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(C.mag_abs(self._ptr))

    def abs_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(C.mag_abs_(self._ptr))

    def neg(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(C.mag_neg(self._ptr))

    def neg_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(C.mag_neg_(self._ptr))

    def __neg__(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return self.neg()

    def __round__(self, n: int | None = None) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return self.round()

    def log(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(C.mag_log(self._ptr))

    def log_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(C.mag_log_(self._ptr))

    def sqr(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(C.mag_sqr(self._ptr))

    def sqr_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(C.mag_sqr_(self._ptr))

    def sqrt(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(C.mag_sqrt(self._ptr))

    def sqrt_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(C.mag_sqrt_(self._ptr))

    def sin(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(C.mag_sin(self._ptr))

    def sin_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(C.mag_sin_(self._ptr))

    def cos(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(C.mag_cos(self._ptr))

    def cos_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(C.mag_cos_(self._ptr))

    def step(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(C.mag_step(self._ptr))

    def step_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(C.mag_step_(self._ptr))

    def exp(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(C.mag_exp(self._ptr))

    def exp_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(C.mag_exp_(self._ptr))

    def floor(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(C.mag_floor(self._ptr))

    def floor_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(C.mag_floor_(self._ptr))

    def ceil(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(C.mag_ceil(self._ptr))

    def ceil_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(C.mag_ceil_(self._ptr))

    def round(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(C.mag_round(self._ptr))

    def round_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(C.mag_round_(self._ptr))

    def softmax(self, dim: int = -1) -> Tensor:
        if dim != -1:
            raise NotImplementedError('Softmax only supports the last dimension (-1) for now')
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(C.mag_softmax(self._ptr))

    def softmax_(self, dim: int = -1) -> Tensor:
        if dim != -1:
            raise NotImplementedError('Softmax only supports the last dimension (-1) for now')
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(C.mag_softmax_(self._ptr))

    def sigmoid(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(C.mag_sigmoid(self._ptr))

    def sigmoid_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(C.mag_sigmoid_(self._ptr))

    def hardsigmoid(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(C.mag_hard_sigmoid(self._ptr))

    def hardsigmoid_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(C.mag_hard_sigmoid_(self._ptr))

    def silu(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(C.mag_silu(self._ptr))

    def silu_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(C.mag_silu_(self._ptr))

    def tanh(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(C.mag_tanh(self._ptr))

    def tanh_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(C.mag_tanh_(self._ptr))

    def relu(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(C.mag_relu(self._ptr))

    def relu_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(C.mag_relu_(self._ptr))

    def gelu(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(C.mag_gelu(self._ptr))

    def gelu_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(C.mag_gelu_(self._ptr))

    def gelu_approx(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(C.mag_gelu_approx(self._ptr))

    def gelu_approx_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(C.mag_gelu_approx_(self._ptr))

    def tril(self, diagonal: int = 0) -> Tensor:
        assert self.rank >= 2, f'Tril requires a rank >= 2 but is {self.rank}'
        return Tensor(C.mag_tril(self._ptr, diagonal))

    def tril_(self, diagonal: int = 0) -> Tensor:
        assert self.rank >= 2, f'Tril requires a rank >= 2 but is {self.rank}'
        self._validate_inplace_op()
        return Tensor(C.mag_tril_(self._ptr, diagonal))

    def triu(self, diagonal: int = 0) -> Tensor:
        assert self.rank >= 2, f'Triu requires a rank >= 2 but is {self.rank}'
        return Tensor(C.mag_triu(self._ptr, diagonal))

    def triu_(self, diagonal: int = 0) -> Tensor:
        assert self.rank >= 2, f'Triu requires a rank >= 2 but is {self.rank}'
        self._validate_inplace_op()
        return Tensor(C.mag_triu_(self._ptr, diagonal))

    def multinomial(self, num_samples: int = 1, replacement: bool = False) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        assert self.rank in (1, 2), f'Multinomial sampling requires a 1D or 2D tensor, but got rank {self.rank}'
        assert num_samples > 0
        assert not replacement, 'Multinomial sampling with replacement is not implemented yet'
        return Tensor(C.mag_multinomial(self._ptr, num_samples, replacement))

    def logical_and(self, rhs: Tensor) -> Tensor:
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(C.mag_and(self._ptr, rhs._ptr))

    def logical_and_(self, rhs: Tensor) -> Tensor:
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(C.mag_and_(self._ptr, rhs._ptr))

    def logical_or(self, rhs: Tensor) -> Tensor:
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(C.mag_and(self._ptr, rhs._ptr))

    def logical_or_(self, rhs: Tensor) -> Tensor:
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(C.mag_or_(self._ptr, rhs._ptr))

    def logical_xor(self, rhs: Tensor) -> Tensor:
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(C.mag_and(self._ptr, rhs._ptr))

    def logical_xor_(self, rhs: Tensor) -> Tensor:
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(C.mag_xor_(self._ptr, rhs._ptr))

    def logical_not(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=INTEGRAL_DTYPES)
        return Tensor(C.mag_not(self._ptr))

    def logical_not_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=INTEGRAL_DTYPES)
        self._validate_inplace_op()
        return Tensor(C.mag_not_(self._ptr))

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
        return Tensor(C.mag_add(self._ptr, rhs._ptr))

    def __radd__(self, rhs: int | float) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return rhs + self

    def __iadd__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(C.mag_add_(self._ptr, float(rhs)))

    def __sub__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(C.mag_sub(self._ptr, rhs._ptr))

    def __rsub__(self, rhs: int | float) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return rhs - self

    def __isub__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(C.mag_sub_(self._ptr, rhs._ptr))

    def __mul__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(C.mag_mul(self._ptr, rhs._ptr))

    def __rmul__(self, rhs: int | float) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return rhs * self

    def __imul__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(C.mag_mul_(self._ptr, rhs._ptr))

    def __truediv__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(C.mag_div(self._ptr, rhs._ptr))

    def __rtruediv__(self, rhs: int | float) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return rhs / self

    def __itruediv__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(C.mag_div_(self._ptr, rhs._ptr))

    def __floordiv__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(C.mag_div(self._ptr, rhs._ptr))

    def __rfloordiv__(self, rhs: int | float) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return rhs / self

    def __ifloordiv__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(C.mag_div_(self._ptr, rhs._ptr))

    def __matmul__(self, rhs: Tensor) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(C.mag_matmul(self._ptr, rhs._ptr))

    def __imatmul__(self, rhs: Tensor) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(C.mag_matmul_(self._ptr, rhs._ptr))

    def __and__(self, rhs: Tensor | bool | int) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(C.mag_and(self._ptr, rhs._ptr))

    def __rand__(self, rhs: int | bool) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return rhs & self

    def __iand__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(C.mag_and_(self._ptr, rhs._ptr))

    def __or__(self, rhs: Tensor | bool | int) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(C.mag_or(self._ptr, rhs._ptr))

    def __ror__(self, rhs: int | bool) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return rhs | self

    def __ior__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(C.mag_or_(self._ptr, rhs._ptr))

    def __xor__(self, rhs: Tensor | bool | int) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(C.mag_xor(self._ptr, rhs._ptr))

    def __rxor__(self, rhs: int | bool) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return rhs ^ self

    def __ixor__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(C.mag_xor_(self._ptr, rhs._ptr))

    def __invert__(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=INTEGRAL_DTYPES)
        return Tensor(C.mag_not(self._ptr))

    def __lshift__(self, rhs: Tensor | int) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(C.mag_shl(self._ptr, rhs._ptr))

    def __rlshift__(self, rhs: int) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return rhs << self

    def __ilshift__(self, rhs: Tensor | int) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(C.mag_shl_(self._ptr, rhs._ptr))

    def __rshift__(self, rhs: Tensor | int) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(C.mag_shr(self._ptr, rhs._ptr))

    def __rrshift__(self, rhs: int) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return rhs >> self

    def __irshift__(self, rhs: Tensor | int) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(C.mag_shr_(self._ptr, rhs._ptr))

    def __eq__(self, rhs: Tensor | list[int | float | bool] | int | float | bool) -> Tensor:
        rhs = self._expand_rhs_list(rhs)
        return Tensor(C.mag_eq(self._ptr, rhs._ptr))

    def __ne__(self, rhs: Tensor | list[int | float | bool] | int | float | bool) -> Tensor:
        rhs = self._expand_rhs_list(rhs)
        return Tensor(C.mag_ne(self._ptr, rhs._ptr))

    def __le__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(C.mag_le(self._ptr, rhs._ptr))

    def __ge__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(C.mag_ge(self._ptr, rhs._ptr))

    def __lt__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(C.mag_lt(self._ptr, rhs._ptr))

    def __gt__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(C.mag_gt(self._ptr, rhs._ptr))
