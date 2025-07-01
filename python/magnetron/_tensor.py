# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import threading
import weakref

from ._context import Context, default_dtype
from ._core import *
from ._bootstrap import load_native_module

_ffi, _C = load_native_module()

_MAIN_TID: int = threading.get_native_id()


def _deduce_tensor_dtype(obj: bool | float | int) -> DataType:
    if isinstance(obj, bool):
        return boolean
    elif isinstance(obj, int):
        return int32
    elif isinstance(obj, float):
        return float32
    else:
        raise TypeError(f'Invalid data type: {type(obj)}')


def get_reduction_axes(dim: int | tuple[int] | None) -> tuple[_ffi.CData, int]:
    if dim is None:
        return _ffi.NULL, 0
    if isinstance(dim, int):
        return _ffi.new('int64_t[1]', [dim]), 1
    elif isinstance(dim, tuple):
        num: int = len(dim)
        axes: _ffi.CData = (_ffi.new(f'int64_t[{num}]', dim),)
        return axes, num
    else:
        raise TypeError('Dimension must be an int or a tuple of ints.')


class Tensor:
    """A 1-6 dimensional tensor with support for automatic differentiation."""

    __slots__ = ('__weakref__', '_ctx', '_ptr', '_finalizer')

    @property
    def name(self) -> str:
        return _ffi.string(_C.mag_tensor_get_name(self._ptr)).decode('utf-8')

    @name.setter
    def name(self, name: str) -> None:
        _C.mag_tensor_set_name(self._ptr, bytes(name, 'utf-8'))

    @property
    def rank(self) -> int:
        return _C.mag_tensor_get_rank(self._ptr)

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(_ffi.unpack(_C.mag_tensor_get_shape(self._ptr), self.rank))

    @property
    def strides(self) -> tuple[int, ...]:
        return tuple(_ffi.unpack(_C.mag_tensor_get_strides(self._ptr), self.rank))

    @property
    def dtype(self) -> DataType:
        dtype_value: int = _C.mag_tensor_get_dtype(self._ptr)
        assert dtype_value in DTYPE_ENUM_MAP, f'Unsupported tensor dtype: {dtype_value}'
        return DTYPE_ENUM_MAP[dtype_value]

    @property
    def data_ptr(self) -> int:
        return int(_ffi.cast('uintptr_t', _C.mag_tensor_get_data_ptr(self._ptr)))

    @property
    def storage_base_ptr(self) -> int:
        return int(_ffi.cast('uintptr_t', _C.mag_tensor_get_storage_base_ptr(self._ptr)))

    def item(self) -> float | int | bool:
        if self.numel != 1:
            raise ValueError('Tensor must have exactly one element to retrieve an item')
        if self.dtype.is_floating_point:
            return float(_C.mag_tensor_get_item_float(self._ptr))
        elif self.dtype == int32:
            return int(_C.mag_tensor_get_item_int(self._ptr))
        elif self.dtype == boolean:
            return bool(_C.mag_tensor_get_item_bool(self._ptr))
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
        unpack_fn = _C.mag_tensor_get_data_as_floats if is_fp else _C.mag_tensor_get_raw_data_as_bytes
        free_fn = _C.mag_tensor_get_data_as_floats_free if is_fp else _C.mag_tensor_get_raw_data_as_bytes_free
        ptr = unpack_fn(self._ptr)
        if not is_fp:
            native: str | None = self.dtype.native_type
            assert native is not None, f'Tensor dtype {self.dtype} does not have a native type'
            ptr = _ffi.cast(f'const {native}*', ptr)
        flat = list(_ffi.unpack(ptr, self.numel))
        free_fn(ptr)
        return build_nested_lists(flat, self.shape, self.strides, offset=0, dim=0)

    @property
    def data_size(self) -> int:
        return _C.mag_tensor_get_data_size(self._ptr)

    @property
    def numel(self) -> int:
        return _C.mag_tensor_get_numel(self._ptr)

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
    def view_base(self) -> Tensor | None:
        if not self.is_view:
            return None
        ptr: _ffi.CData = _C.mag_tensor_get_view_base(self._ptr)
        if ptr is None or ptr == _ffi.NULL:
            return None
        return Tensor(ptr)

    @property
    def view_offset(self) -> int:
        return _C.mag_tensor_get_view_offset(self._ptr)

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
    def native_ptr(self) -> _ffi.CData:
        return self._ptr

    @property
    def is_contiguous(self) -> bool:
        return _C.mag_tensor_is_contiguous(self._ptr)

    @property
    def requires_grad(self) -> bool:
        return _C.mag_tensor_requires_grad(self._ptr)

    @requires_grad.setter
    def requires_grad(self, require: bool) -> None:
        if require and not self.dtype.is_floating_point:
            raise RuntimeError(f'Tensors requiring gradients must be of a floating point type, but is: {self.dtype}')
        _C.mag_tensor_set_requires_grad(self._ptr, require)

    @property
    def grad(self) -> Tensor | None:
        if not self.requires_grad:
            return None
        ptr: _ffi.CData = _C.mag_tensor_get_grad(self._ptr)
        if ptr is None or ptr == _ffi.NULL:
            return None
        return Tensor(ptr)

    def backward(self) -> None:
        assert self.requires_grad, 'Tensor must require gradient tracking'
        assert self.rank == 1 and self.numel == 1, 'Tensor must be scalar'
        _C.mag_tensor_backward(self._ptr)

    def zero_grad(self) -> None:
        assert self.requires_grad, 'Tensor must require gradient tracking'
        _C.mag_tensor_zero_grad(self._ptr)

    def dump_graph_dot(self, file_path: str, forward: bool) -> None:
        file_path = bytes(file_path, 'utf-8')
        if forward:
            _C.mag_tensor_export_forward_graph_graphviz(self._ptr, file_path)
        else:
            _C.mag_tensor_export_backward_graph_graphviz(self._ptr, file_path)

    def __len__(self) -> int:
        return self.shape[0]

    def __str__(self) -> str:
        cstr: _ffi.CData = _C.mag_tensor_to_string(self._ptr, False, 0, 0)
        data_str: str = _ffi.string(cstr).decode('utf-8')
        _C.mag_tensor_to_string_free_data(cstr)
        return data_str

    def __repr__(self) -> str:
        return str(self)

    def __getitem__(self, indices: int | tuple[int, ...]) -> float:
        if isinstance(indices, int):
            return _C.mag_tensor_subscript_get_flattened(self._ptr, indices)
        elif isinstance(indices, tuple):
            idx = indices + (0,) * (MAX_DIMS - len(indices))
            return _C.mag_tensor_subscript_get_multi(self._ptr, *idx)
        else:
            raise TypeError('Indices must be an int or a tuple of ints.')

    def __setitem__(self, indices: int | tuple[int, ...], value: float) -> None:
        if isinstance(indices, int):
            _C.mag_tensor_subscript_set_flattened(self._ptr, indices, float(value))
        elif isinstance(indices, tuple):
            idx = indices + (0,) * (MAX_DIMS - len(indices))
            _C.mag_tensor_subscript_set_multi(self._ptr, *idx, float(value))
        else:
            raise TypeError('Indices must be an int or a tuple of ints.')

    def _validate_inplace_op(self) -> None:
        if Context.get().is_grad_recording and self.requires_grad:
            raise RuntimeError(
                'In-place operations are not allowed when gradient recording is enabled. '
                'Either disable gradient recording or use the `detach()` method to create a new tensor without gradient tracking.'
            )

    def _expand_rhs(self, rhs: Tensor | int | float | bool) -> Tensor:
        if isinstance(rhs, Tensor):
            return rhs
        return Tensor.full_like(self, rhs)

    def _expand_rhs_list(self, rhs: Tensor | int | float | bool | list[int | float | bool]) -> Tensor:
        if isinstance(rhs, list):
            return Tensor.of(rhs, dtype=self.dtype)
        return self._expand_rhs(rhs)

    @staticmethod
    def _validate_dtypes(*args: Tensor, allowed_types: set[DataType]) -> None:
        for i, tensor in enumerate(args):
            if not tensor.dtype in allowed_types:
                raise RuntimeError(f'Operation requires dtype {allowed_types} for arg {i + 1} but got {tensor.dtype}')

    def __init__(self, native_object: _ffi.CData | None) -> None:
        assert _MAIN_TID == threading.get_native_id(), 'Context must be created in the main thread'
        self._ctx = Context.get()
        self._ptr = native_object
        self._finalizer = weakref.finalize(self, _C.mag_tensor_decref, self._ptr)

    @classmethod
    def empty(
        cls,
        *shape: int | tuple[int, ...],
        dtype: DataType = default_dtype(),
        requires_grad: bool = False,
        name: str | None = None,
    ) -> Tensor:
        shape: tuple[int, ...] = unpack_shape(*shape)
        assert 0 < len(shape) <= MAX_DIMS, f'Invalid number of dimensions: {len(shape)}'
        assert all(0 < dim <= DIM_MAX for dim in shape), 'Invalid dimension size'
        dims: _ffi.CData = _ffi.new(f'int64_t[{len(shape)}]', shape)
        instance: _ffi.CData = _C.mag_tensor_empty(Context.get().native_ptr, dtype.enum_value, len(shape), dims)
        tensor: Tensor = cls(instance)
        tensor.requires_grad = requires_grad
        if name is not None:
            tensor.name = name
        return tensor

    @classmethod
    def empty_like(
        cls,
        template: Tensor,
        *,
        dtype: DataType = default_dtype(),
        requires_grad: bool = False,
        name: str | None = None,
    ) -> Tensor:
        return cls.empty(template.shape, dtype=dtype, requires_grad=requires_grad, name=name)

    @classmethod
    def full(
        cls,
        *shape: int | tuple[int, ...],
        fill_value: int | float | bool,
        dtype: DataType = default_dtype(),
        requires_grad: bool = False,
        name: str | None = None,
    ) -> Tensor:
        shape: tuple[int, ...] = unpack_shape(*shape)
        tensor: Tensor = cls.empty(
            *shape,
            dtype=dtype,
            requires_grad=requires_grad,
            name=name,
        )
        tensor.fill_(fill_value)
        return tensor

    @classmethod
    def full_like(
        cls,
        template: Tensor,
        fill_value: int | float | bool,
        *,
        dtype: DataType = default_dtype(),
        requires_grad: bool = False,
        name: str | None = None,
    ) -> Tensor:
        return cls.full(
            template.shape,
            fill_value=fill_value,
            dtype=dtype,
            requires_grad=requires_grad,
            name=name,
        )

    @classmethod
    def of(
        cls,
        data: NestedList,
        *,
        dtype: DataType | None = None,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> Tensor:
        if not data:
            return cls.empty(0, dtype=dtype if dtype is not None else default_dtype())
        shape, flattened_data = flatten_nested_lists(data)
        dtype: DataType = dtype if dtype is not None else _deduce_tensor_dtype(flattened_data[0])
        native_name: str = dtype.native_type
        alloc_fn: _ffi.CData = dtype.fill_fn
        tensor: Tensor = cls.empty(
            *shape,
            dtype=dtype,
            requires_grad=requires_grad,
            name=name,
        )
        staging_buffer: _ffi.CData = _ffi.new(f'{native_name}[{len(flattened_data)}]', flattened_data)
        copy_bytes_numel: int = len(flattened_data)
        if (
            alloc_fn == _C.mag_tensor_fill_from_raw_bytes
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
        name: str | None = None,
    ) -> Tensor:
        return cls.full(*shape, fill_value=0, dtype=dtype, requires_grad=requires_grad, name=name)

    @classmethod
    def zeros_like(
        cls,
        template: Tensor,
        dtype: DataType = default_dtype(),
        *,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> Tensor:
        return cls.zeros(template.shape, dtype=dtype, requires_grad=requires_grad, name=name)

    @classmethod
    def ones(
        cls,
        *shape: int | tuple[int, ...],
        dtype: DataType = default_dtype(),
        requires_grad: bool = False,
        name: str | None = None,
    ) -> Tensor:
        return cls.full(*shape, fill_value=1, dtype=dtype, requires_grad=requires_grad, name=name)

    @classmethod
    def ones_like(
        cls,
        template: Tensor,
        *,
        dtype: DataType = default_dtype(),
        requires_grad: bool = False,
        name: str | None = None,
    ) -> Tensor:
        return cls.ones(template.shape, dtype=dtype, requires_grad=requires_grad, name=name)

    @classmethod
    def uniform(
        cls,
        *shape: int | tuple[int, ...],
        low: float | int | None = None,
        high: float | int | None = None,
        dtype: DataType = default_dtype(),
        requires_grad: bool = False,
        name: str | None = None,
    ) -> Tensor:
        shape: tuple[int, ...] = unpack_shape(*shape)
        tensor: Tensor = cls.empty(
            *shape,
            dtype=dtype,
            requires_grad=requires_grad,
            name=name,
        )
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
        name: str | None = None,
    ) -> Tensor:
        shape: tuple[int, ...] = unpack_shape(*shape)
        tensor: Tensor = cls.empty(
            *shape,
            dtype=dtype,
            requires_grad=requires_grad,
            name=name,
        )
        tensor.fill_random_normal_(mean, std)
        return tensor

    @classmethod
    def bernoulli(
        cls,
        *shape: int | tuple[int, ...],
        p: float = 0.5,
        name: str | None = None,
    ) -> Tensor:
        shape: tuple[int, ...] = unpack_shape(*shape)
        tensor: Tensor = cls.empty(
            *shape,
            dtype=boolean,
            requires_grad=False,
            name=name,
        )
        tensor.fill_random_bernoulli_(p)
        return tensor

    def clone(self) -> Tensor:
        return Tensor(_C.mag_clone(self._ptr))

    def view(self, *dims: int | tuple[int, ...]) -> Tensor:
        dims = unpack_shape(dims)
        assert self.is_contiguous, 'Tensor must be contiguous to be viewed'
        num_dims: int = len(dims)
        view_dims: _ffi.CData = _ffi.new(f'int64_t[{num_dims}]', dims)
        return Tensor(_C.mag_view(self._ptr, view_dims, num_dims))

    def reshape(self, *dims: int | tuple[int, ...]) -> Tensor:
        dims = unpack_shape(dims)
        num_dims: int = len(dims)
        view_dims: _ffi.CData = _ffi.new(f'int64_t[{num_dims}]', dims)
        return Tensor(_C.mag_reshape(self._ptr, view_dims, num_dims))

    def transpose(self, dim1: int = 0, dim2: int = 1) -> Tensor:
        assert dim1 != dim2, f'Transposition axes must be not equal, but {dim1} == {dim2}'
        return Tensor(_C.mag_transpose(self._ptr, dim1, dim2))

    @property
    def T(self) -> Tensor:
        return Tensor(_C.mag_transpose(self._ptr, 0, 1))

    def detach(self) -> Tensor:
        _C.mag_tensor_detach(self._ptr)
        return self

    def contiguous(self) -> Tensor:
        if self.is_contiguous:
            return self
        return self.clone()

    def permute(self, *dims: int | tuple[int, ...]) -> Tensor:
        dims = unpack_shape(*dims)
        assert len(dims) == self.rank, f'Invalid number of axes, require {self.rank}, got {len(dims)}'
        if len(dims) != MAX_DIMS:
            dims = dims + tuple(range(self.rank, MAX_DIMS))
        assert len(dims) == MAX_DIMS
        dims = _ffi.new('int64_t[]', dims)
        for i in range(MAX_DIMS):
            assert 0 <= dims[i] < MAX_DIMS
            for j in range(i + 1, MAX_DIMS):
                assert dims[i] != dims[j], f'Duplicate axis: {dims[i]}'
        return Tensor(_C.mag_permute(self._ptr, dims, MAX_DIMS))

    def fill_(self, value: float | int | bool) -> None:
        self._validate_inplace_op()
        if self.dtype.is_floating_point:
            _C.mag_tensor_fill_float(self._ptr, float(value))
        else:
            _C.mag_tensor_fill_int(self._ptr, int(value))

    def fill_random_uniform_(self, from_: float | int | None = None, to: float | int | None = None) -> None:
        self._validate_dtypes(self, allowed_types=NUMERIC_DTYPES)
        self._validate_inplace_op()
        from_def, to_def = (-0x80000000, 0x7FFFFFFF) if self.dtype.is_integer else (0.0, 1.0)
        from_ = from_def if from_ is None else from_
        to = to_def if to is None else to
        assert to > from_, f'Invalid uniform range {to} must be > {from_}'
        if self.dtype.is_floating_point:
            _C.mag_tensor_fill_random_uniform_float(self._ptr, float(from_), float(to))
        else:
            _C.mag_tensor_fill_random_uniform_int(self._ptr, int(from_), int(to))

    def fill_random_normal_(self, mean: float, std: float) -> None:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        _C.mag_tensor_fill_random_normal(self._ptr, mean, std)

    def fill_random_bernoulli_(self, p: float) -> None:
        self._validate_dtypes(self, allowed_types={boolean})
        self._validate_inplace_op()
        _C.mag_tensor_fill_random_bernoulli(self._ptr, p)

    def zeros_(self) -> None:
        self.fill_(0)

    def ones_(self) -> None:
        self.fill_(1)

    def mean(self, dim: int | tuple[int] | None = None, keepdim: bool = False) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        dims, num_dims = get_reduction_axes(dim)
        return Tensor(_C.mag_mean(self._ptr, dims, num_dims, keepdim))

    def min(self, dim: int | tuple[int] | None = None, keepdim: bool = False) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        dims, num_dims = get_reduction_axes(dim)
        return Tensor(_C.mag_min(self._ptr, dims, num_dims, keepdim))

    def max(self, dim: int | tuple[int] | None = None, keepdim: bool = False) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        dims, num_dims = get_reduction_axes(dim)
        return Tensor(_C.mag_max(self._ptr, dims, num_dims, keepdim))

    def sum(self, dim: int | tuple[int] | None = None, keepdim: bool = False) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        dims, num_dims = get_reduction_axes(dim)
        return Tensor(_C.mag_sum(self._ptr, dims, num_dims, keepdim))

    def argmin(self, dim: int | tuple[int] | None = None, keepdim: bool = False) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        dims, num_dims = get_reduction_axes(dim)
        return Tensor(_C.mag_argmin(self._ptr, dims, num_dims, keepdim))

    def argmax(self, dim: int | tuple[int] | None = None, keepdim: bool = False) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        dims, num_dims = get_reduction_axes(dim)
        return Tensor(_C.mag_argmax(self._ptr, dims, num_dims, keepdim))

    def abs(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_C.mag_abs(self._ptr))

    def abs_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_C.mag_abs_(self._ptr))

    def neg(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_C.mag_neg(self._ptr))

    def neg_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_C.mag_neg_(self._ptr))

    def __neg__(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return self.neg()

    def log(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_C.mag_log(self._ptr))

    def log_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_C.mag_log_(self._ptr))

    def sqr(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_C.mag_sqr(self._ptr))

    def sqr_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_C.mag_sqr_(self._ptr))

    def sqrt(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_C.mag_sqrt(self._ptr))

    def sqrt_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_C.mag_sqrt_(self._ptr))

    def sin(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_C.mag_sin(self._ptr))

    def sin_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_C.mag_sin_(self._ptr))

    def cos(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_C.mag_cos(self._ptr))

    def cos_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_C.mag_cos_(self._ptr))

    def step(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_C.mag_step(self._ptr))

    def step_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_C.mag_step_(self._ptr))

    def exp(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_C.mag_exp(self._ptr))

    def exp_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_C.mag_exp_(self._ptr))

    def floor(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_C.mag_floor(self._ptr))

    def floor_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_C.mag_floor_(self._ptr))

    def ceil(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_C.mag_ceil(self._ptr))

    def ceil_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_C.mag_ceil_(self._ptr))

    def round(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_C.mag_round(self._ptr))

    def round_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_C.mag_round_(self._ptr))

    def softmax(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_C.mag_softmax(self._ptr))

    def softmax_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_C.mag_softmax_(self._ptr))

    def sigmoid(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_C.mag_sigmoid(self._ptr))

    def sigmoid_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_C.mag_sigmoid_(self._ptr))

    def hardsigmoid(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_C.mag_hard_sigmoid(self._ptr))

    def hardsigmoid_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_C.mag_hard_sigmoid_(self._ptr))

    def silu(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_C.mag_silu(self._ptr))

    def silu_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_C.mag_silu_(self._ptr))

    def tanh(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_C.mag_tanh(self._ptr))

    def tanh_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_C.mag_tanh_(self._ptr))

    def relu(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_C.mag_relu(self._ptr))

    def relu_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_C.mag_relu_(self._ptr))

    def gelu(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_C.mag_gelu(self._ptr))

    def gelu_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_C.mag_gelu_(self._ptr))

    def tril(self, diagonal: int = 0) -> Tensor:
        assert self.rank >= 2, f'Tril requires a rank >= 2 but is {self.rank}'
        return Tensor(_C.mag_tril(self._ptr, diagonal))

    def tril_(self, diagonal: int = 0) -> Tensor:
        assert self.rank >= 2, f'Tril requires a rank >= 2 but is {self.rank}'
        self._validate_inplace_op()
        return Tensor(_C.mag_tril_(self._ptr, diagonal))

    def triu(self, diagonal: int = 0) -> Tensor:
        assert self.rank >= 2, f'Triu requires a rank >= 2 but is {self.rank}'
        return Tensor(_C.mag_triu(self._ptr, diagonal))

    def triu_(self, diagonal: int = 0) -> Tensor:
        assert self.rank >= 2, f'Triu requires a rank >= 2 but is {self.rank}'
        self._validate_inplace_op()
        return Tensor(_C.mag_triu_(self._ptr, diagonal))

    def logical_and(self, rhs: Tensor) -> Tensor:
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_C.mag_and(self._ptr, rhs._ptr))

    def logical_and_(self, rhs: Tensor) -> Tensor:
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_C.mag_and_(self._ptr, rhs._ptr))

    def logical_or(self, rhs: Tensor) -> Tensor:
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_C.mag_and(self._ptr, rhs._ptr))

    def logical_or_(self, rhs: Tensor) -> Tensor:
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_C.mag_or_(self._ptr, rhs._ptr))

    def logical_xor(self, rhs: Tensor) -> Tensor:
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_C.mag_and(self._ptr, rhs._ptr))

    def logical_xor_(self, rhs: Tensor) -> Tensor:
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_C.mag_xor_(self._ptr, rhs._ptr))

    def logical_not(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_C.mag_not(self._ptr))

    def logical_not_(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=INTEGRAL_DTYPES)
        self._validate_inplace_op()
        return Tensor(_C.mag_not_(self._ptr))

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
        return Tensor(_C.mag_add(self._ptr, rhs._ptr))

    def __radd__(self, rhs: int | float) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return rhs + self

    def __iadd__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_C.mag_add_(self._ptr, float(rhs)))

    def __sub__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_C.mag_sub(self._ptr, rhs._ptr))

    def __rsub__(self, rhs: int | float) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return rhs - self

    def __isub__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_C.mag_sub_(self._ptr, rhs._ptr))

    def __mul__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_C.mag_mul(self._ptr, rhs._ptr))

    def __rmul__(self, rhs: int | float) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return rhs * self

    def __imul__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_C.mag_mul_(self._ptr, rhs._ptr))

    def __truediv__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_C.mag_div(self._ptr, rhs._ptr))

    def __rtruediv__(self, rhs: int | float) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return rhs / self

    def __itruediv__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_C.mag_div_(self._ptr, rhs._ptr))

    def __floordiv__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_C.mag_div(self._ptr, rhs._ptr))

    def __rfloordiv__(self, rhs: int | float) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return rhs / self

    def __ifloordiv__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_C.mag_div_(self._ptr, rhs._ptr))

    def __matmul__(self, rhs: Tensor) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        return Tensor(_C.mag_matmul(self._ptr, rhs._ptr))

    def __imatmul__(self, rhs: Tensor) -> Tensor:
        self._validate_dtypes(self, allowed_types=FLOATING_POINT_DTYPES)
        self._validate_inplace_op()
        return Tensor(_C.mag_matmul_(self._ptr, rhs._ptr))

    def __and__(self, rhs: Tensor | bool | int) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_C.mag_and(self._ptr, rhs._ptr))

    def __rand__(self, rhs: int | bool) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return rhs & self

    def __iand__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_C.mag_and_(self._ptr, rhs._ptr))

    def __or__(self, rhs: Tensor | bool | int) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_C.mag_or(self._ptr, rhs._ptr))

    def __ror__(self, rhs: int | bool) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return rhs | self

    def __ior__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_C.mag_or_(self._ptr, rhs._ptr))

    def __xor__(self, rhs: Tensor | bool | int) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_C.mag_xor(self._ptr, rhs._ptr))

    def __rxor__(self, rhs: int | bool) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return rhs ^ self

    def __ixor__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_C.mag_xor_(self._ptr, rhs._ptr))

    def __invert__(self) -> Tensor:
        self._validate_dtypes(self, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_C.mag_not(self._ptr))

    def __lshift__(self, rhs: Tensor | int) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_C.mag_shl(self._ptr, rhs._ptr))

    def __rlshift__(self, rhs: int) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return rhs << self

    def __ilshift__(self, rhs: Tensor | int) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_C.mag_shl_(self._ptr, rhs._ptr))

    def __rshift__(self, rhs: Tensor | int) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_C.mag_shr(self._ptr, rhs._ptr))

    def __rrshift__(self, rhs: int) -> Tensor:
        rhs = Tensor.full_like(self, rhs)
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return rhs >> self

    def __irshift__(self, rhs: Tensor | int) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_inplace_op()
        self._validate_dtypes(self, rhs, allowed_types=INTEGRAL_DTYPES)
        return Tensor(_C.mag_shr_(self._ptr, rhs._ptr))

    def __eq__(self, rhs: Tensor | list[int | float | bool] | int | float | bool) -> Tensor:
        rhs = self._expand_rhs_list(rhs)
        return Tensor(_C.mag_eq(self._ptr, rhs._ptr))

    def __ne__(self, rhs: Tensor | list[int | float | bool] | int | float | bool) -> Tensor:
        rhs = self._expand_rhs_list(rhs)
        return Tensor(_C.mag_ne(self._ptr, rhs._ptr))

    def __le__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_C.mag_le(self._ptr, rhs._ptr))

    def __ge__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_C.mag_ge(self._ptr, rhs._ptr))

    def __lt__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_C.mag_lt(self._ptr, rhs._ptr))

    def __gt__(self, rhs: Tensor | int | float) -> Tensor:
        rhs = self._expand_rhs(rhs)
        self._validate_dtypes(self, rhs, allowed_types=NUMERIC_DTYPES)
        return Tensor(_C.mag_gt(self._ptr, rhs._ptr))
