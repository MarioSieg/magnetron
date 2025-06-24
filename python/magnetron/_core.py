# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import contextlib
import faulthandler
import threading
import typing
import weakref
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum, unique
from functools import lru_cache
from os import getenv
from typing import Optional

from magnetron._bootstrap import load_native_module

faulthandler.enable()

_ffi, _C = load_native_module()

MAX_DIMS: int = 6
DIM_MAX: int = (1 << 63) - 1  # INT64_MAX


class ComputeDevice:
    class CPU:
        def __init__(self, num_threads: int = 0) -> None:
            self.num_threads = num_threads

    class CUDA:
        def __init__(self, device_id: int = 0) -> None:
            self.device_id = device_id


@unique
class PRNGAlgorithm(Enum):
    MERSENNE_TWISTER = _C.MAG_PRNG_MERSENNE_TWISTER
    PCG = _C.MAG_PRNG_PCG


@dataclass(frozen=True)
class DataType:
    enum_value: int
    size: int
    name: str
    native_type: str | None

    def __str__(self) -> str:
        return self.name

    def __repr__(self) -> str:
        return self.name


float32: DataType = DataType(_C.MAG_DTYPE_E8M23, 4, 'float32', 'float')
float16: DataType = DataType(_C.MAG_DTYPE_E5M10, 2, 'float16', None)
boolean: DataType = DataType(_C.MAG_DTYPE_BOOL, 1, 'bool', 'bool')
int32: DataType = DataType(_C.MAG_DTYPE_I32, 4, 'int32', 'int32_t')

_DTYPE_ENUM_MAP: dict[int, DataType] = {
    float32.enum_value: float32,
    float16.enum_value: float16,
    boolean.enum_value: boolean,
    int32.enum_value: int32,
}

# Includes all floating-point types.
_FLOATING_POINT_DTYPES: set[DataType] = {float32, float16}

# Includes all integral types (integers + boolean).
_INTEGRAL_DTYPES: set[DataType] = {boolean, int32}

# Include all integer types (integers - boolean).
_INTEGER_DTYPES: set[DataType] = _INTEGRAL_DTYPES - {boolean}

# Include all numeric dtypes (floating point + integers - boolean)
_NUMERIC_DTYPES = _FLOATING_POINT_DTYPES | _INTEGER_DTYPES


@dataclass
class Config:
    verbose: bool = getenv('MAGNETRON_VERBOSE', '0') == '1'
    compute_device: ComputeDevice.CPU | ComputeDevice.CUDA = ComputeDevice.CPU()
    default_dtype: DataType = float32


_MAIN_TID: int = threading.get_native_id()


@typing.final
class Context:
    """Manages the execution context and owns all tensors and active compute devices."""

    @staticmethod
    @lru_cache(maxsize=1)
    def primary() -> 'Context':
        """Get global context singleton."""
        _C.mag_set_log_mode(Config.verbose)
        return Context()

    def __init__(self, device: ComputeDevice.CPU | ComputeDevice.CUDA = Config.compute_device) -> None:
        assert _MAIN_TID == threading.get_native_id(), 'Context must be created in the main thread'
        desc: _ffi.CData = _ffi.new('mag_ComputeDeviceDesc*')
        if isinstance(device, ComputeDevice.CPU):
            desc[0] = _C.mag_compute_device_desc_cpu(device.num_threads)
        elif isinstance(device, ComputeDevice.CUDA):
            desc[0] = _C.mag_compute_device_desc_cuda(device.device_id)
        self._ptr = _C.mag_ctx_create2(desc)
        self.default_dtype = Config.default_dtype
        self._finalizer = weakref.finalize(self, _C.mag_ctx_destroy, self._ptr)

    @property
    def native_ptr(self) -> _ffi.CData:
        return self._ptr

    @property
    def compute_device_name(self) -> str:
        return _ffi.string(_C.mag_ctx_get_compute_device_name(self._ptr)).decode('utf-8')

    @property
    def prng_algorithm(self) -> PRNGAlgorithm:
        return PRNGAlgorithm(_C.mag_ctx_get_prng_algorithm(self._ptr))

    @prng_algorithm.setter
    def prng_algorithm(self, algorithm: PRNGAlgorithm) -> None:
        _C.mag_ctx_set_prng_algorithm(self._ptr, algorithm.value, 0)

    def seed(self, seed: int) -> None:
        _C.mag_ctx_set_prng_algorithm(self._ptr, self.prng_algorithm.value, seed)

    @property
    def os_name(self) -> str:
        return _ffi.string(_C.mag_ctx_get_os_name(self._ptr)).decode('utf-8')

    @property
    def cpu_name(self) -> str:
        return _ffi.string(_C.mag_ctx_get_cpu_name(self._ptr)).decode('utf-8')

    @property
    def cpu_virtual_cores(self) -> int:
        return _C.mag_ctx_get_cpu_virtual_cores(self._ptr)

    @property
    def cpu_physical_cores(self) -> int:
        return _C.mag_ctx_get_cpu_physical_cores(self._ptr)

    @property
    def cpu_sockets(self) -> int:
        return _C.mag_ctx_get_cpu_sockets(self._ptr)

    @property
    def physical_memory_total(self) -> int:
        return _C.mag_ctx_get_physical_memory_total(self._ptr)

    @property
    def physical_memory_free(self) -> int:
        return _C.mag_ctx_get_physical_memory_free(self._ptr)

    @property
    def physical_memory_used(self) -> int:
        return abs(self.physical_memory_total - self.physical_memory_free)

    @property
    def is_numa_system(self) -> bool:
        return _C.mag_ctx_is_numa_system(self._ptr)

    @property
    def is_profiling(self) -> bool:
        return _C.mag_ctx_profiler_is_running(self._ptr)

    def start_grad_recorder(self) -> None:
        _C.mag_ctx_grad_recorder_start(self._ptr)

    def stop_grad_recorder(self) -> None:
        _C.mag_ctx_grad_recorder_stop(self._ptr)

    @property
    def is_grad_recording(self) -> bool:
        return _C.mag_ctx_grad_recorder_is_running(self._ptr)


class no_grad(contextlib.ContextDecorator):
    """Disables gradient recording within a function or block."""

    def __enter__(self) -> None:
        """Disable gradient tracking by stopping the active context's recorder."""
        Context.primary().stop_grad_recorder()

    def __exit__(self, exc_type: any, exc_value: any, traceback: any) -> None:
        """Re-enable gradient tracking when exiting the context."""
        Context.primary().start_grad_recorder()


def _flatten_nested_lists(nested: object) -> tuple[tuple[int, ...], list[float]]:
    """Flatten a nested list and return its shape and flattened data."""
    if not isinstance(nested, list):
        return (), [nested]
    elif len(nested) == 0:
        return (0,), []
    else:
        shapes = []
        flattened = []
        for item in nested:
            shape_lst, flat = _flatten_nested_lists(item)
            shapes.append(shape_lst)
            flattened += flat
        first_shape = shapes[0]
        for s in shapes:
            assert s == first_shape, 'All sub-lists must have the same shape'
        return (len(nested),) + first_shape, flattened


def _deduce_tensor_dtype(obj: any) -> tuple[DataType, str, _ffi.CData]:
    if isinstance(obj, bool):
        return boolean, 'uint8_t', _C.mag_tensor_fill_from_raw_bytes
    elif isinstance(obj, float) or isinstance(obj, int):
        return float32, 'float', _C.mag_tensor_fill_from_floats
    else:
        raise TypeError(f'Invalid data type: {type(obj)}')


def _validate_dtype_compat(dtypes: set[DataType], *kwargs: any) -> None:
    for i, tensor in enumerate(kwargs):
        if tensor.dtype not in dtypes:
            raise TypeError(f'Unsupported data type of argument {i + 1} for operator: {tensor.dtype}')


def _get_reduction_axes(dim: int | tuple[int] | None) -> tuple[_ffi.CData, int]:
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

    def __init__(
        self,
        native_object: _ffi.CData | None = None,
        *,
        ctx: Context | None = None,
        shape: tuple[int, ...] | None = None,
        dtype: DataType = Context.primary().default_dtype,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> None:
        assert _MAIN_TID == threading.get_native_id(), 'Context must be created in the main thread'
        self._ctx = None
        self._ptr = native_object
        if self._ptr is None:  # If no existing native tensor is wrapped, we allocate a new, owned native tensor:
            assert 0 < len(shape) <= MAX_DIMS, f'Invalid number of dimensions: {len(shape)}'
            assert all(0 < dim <= DIM_MAX for dim in shape), 'Invalid dimension size'
            self._ctx = ctx
            dims: _ffi.CData = _ffi.new(f'int64_t[{len(shape)}]', shape)
            self._ptr = _C.mag_tensor_empty(ctx._ptr, dtype.enum_value, len(shape), dims)
            self.requires_grad = requires_grad
            if name is not None:
                self.name = name
        self._finalizer = weakref.finalize(self, _C.mag_tensor_decref, self._ptr)

    @property
    def native_ptr(self) -> _ffi.CData:
        return self._ptr

    @classmethod
    def empty(
        cls,
        shape: tuple[int, ...],
        *,
        dtype: DataType = Context.primary().default_dtype,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> 'Tensor':
        tensor = cls(
            native_object=None,
            ctx=Context.primary(),
            shape=shape,
            dtype=dtype,
            requires_grad=requires_grad,
            name=name,
        )
        return tensor

    @classmethod
    def empty_like(
        cls,
        template: 'Tensor',
        *,
        dtype: DataType = Context.primary().default_dtype,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> 'Tensor':
        return cls.empty(template.shape, dtype=dtype, requires_grad=requires_grad, name=name)

    @classmethod
    def full(
        cls,
        shape: tuple[int, ...],
        *,
        fill_value: int | float,
        dtype: DataType = Context.primary().default_dtype,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> 'Tensor':
        tensor = cls(
            native_object=None,
            ctx=Context.primary(),
            shape=shape,
            dtype=dtype,
            requires_grad=requires_grad,
            name=name,
        )
        tensor.fill_(fill_value)
        return tensor

    @classmethod
    def full_like(
        cls,
        template: 'Tensor',
        *,
        fill_value: int | float,
        dtype: DataType = Context.primary().default_dtype,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> 'Tensor':
        return cls.full(
            template.shape,
            fill_value=fill_value,
            dtype=dtype,
            requires_grad=requires_grad,
            name=name,
        )

    @classmethod
    def from_data(
        cls,
        data: list[float, ...] | list[bool, ...],
        *,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> 'Tensor':
        if not data:
            return cls.empty(shape=(0,), dtype=float32)

        shape, flattened_data = _flatten_nested_lists(data)
        dtype, native_name, alloc_fn = _deduce_tensor_dtype(flattened_data[0])
        tensor = cls(
            native_object=None,
            ctx=Context.primary(),
            shape=shape,
            dtype=dtype,
            requires_grad=requires_grad,
            name=name,
        )
        staging_buffer: _ffi.CData = _ffi.new(f'{native_name}[{len(flattened_data)}]', flattened_data)
        alloc_fn(tensor._ptr, staging_buffer, len(flattened_data))
        return tensor

    @classmethod
    def zeros(
        cls,
        shape: tuple[int, ...],
        *,
        dtype: DataType = Context.primary().default_dtype,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> 'Tensor':
        return cls.full(shape, fill_value=0, dtype=dtype, requires_grad=requires_grad, name=name)

    @classmethod
    def zeros_like(
        cls,
        template: 'Tensor',
        *,
        dtype: DataType = Context.primary().default_dtype,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> 'Tensor':
        return cls.zeros(template.shape, dtype=dtype, requires_grad=requires_grad, name=name)

    @classmethod
    def ones(
        cls,
        shape: tuple[int, ...],
        *,
        dtype: DataType = Context.primary().default_dtype,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> 'Tensor':
        return cls.full(shape, fill_value=1, dtype=dtype, requires_grad=requires_grad, name=name)

    @classmethod
    def ones_like(
        cls,
        template: 'Tensor',
        *,
        dtype: DataType = Context.primary().default_dtype,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> 'Tensor':
        return cls.ones(template.shape, dtype=dtype, requires_grad=requires_grad, name=name)

    @classmethod
    def uniform(
        cls,
        shape: tuple[int, ...],
        *,
        from_: float | int | None = None,
        to: float | int | None = None,
        dtype: DataType = Context.primary().default_dtype,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> 'Tensor':
        tensor = cls(
            native_object=None,
            ctx=Context.primary(),
            shape=shape,
            dtype=dtype,
            requires_grad=requires_grad,
            name=name,
        )
        tensor.fill_random_uniform_(from_, to)
        return tensor

    @classmethod
    def normal(
        cls,
        shape: tuple[int, ...],
        *,
        mean: float = 0.0,
        std: float = 1.0,
        dtype: DataType = Context.primary().default_dtype,
        requires_grad: bool = False,
        name: str | None = None,
    ) -> 'Tensor':
        tensor = cls(
            native_object=None,
            ctx=Context.primary(),
            shape=shape,
            dtype=dtype,
            requires_grad=requires_grad,
            name=name,
        )
        tensor.fill_random_normal_(mean, std)
        return tensor

    @classmethod
    def bernoulli(
        cls,
        shape: tuple[int, ...],
        *,
        p: float = 0.5,
        name: str | None = None,
    ) -> 'Tensor':
        tensor = cls(
            native_object=None,
            ctx=Context.primary(),
            shape=shape,
            dtype=boolean,
            requires_grad=False,
            name=name,
        )
        tensor.fill_random_bernoulli_(p)
        return tensor

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
        assert dtype_value in _DTYPE_ENUM_MAP, f'Unsupported tensor dtype: {dtype_value}'
        return _DTYPE_ENUM_MAP[dtype_value]

    @property
    def data_ptr(self) -> int:
        return int(_ffi.cast('uintptr_t', _C.mag_tensor_get_data_ptr(self._ptr)))

    @property
    def storage_base_ptr(self) -> int:
        return int(_ffi.cast('uintptr_t', _C.mag_tensor_get_storage_base_ptr(self._ptr)))

    def item(self) -> float | int | bool:
        if self.numel != 1:
            raise ValueError('Tensor must have exactly one element to retrieve an item')
        if self.is_floating_point:
            return float(_C.mag_tensor_get_item_float(self._ptr))
        elif self.dtype == int32:
            return int(_C.mag_tensor_get_item_int(self._ptr))
        elif self.dtype == boolean:
            return bool(_C.mag_tensor_get_item_bool(self._ptr))
        else:
            raise TypeError(f'Unsupported tensor dtype for item retrieval: {self.dtype}')

    def __bool__(self) -> bool:
        if self.numel != 1:
            raise ValueError(
                'The truth value of a Tensor with more than one element is '
                'ambiguous. Use .any() or .all() instead.'
            )
        return bool(self.item())

    def tolist(self) -> list[float] | list[int] | list[bool]:
        unpack_fn = _C.mag_tensor_get_data_as_floats if self.is_floating_point else _C.mag_tensor_get_raw_data_as_bytes
        free_fn = _C.mag_tensor_get_data_as_floats_free if self.is_floating_point else _C.mag_tensor_get_raw_data_as_bytes_free
        castor = None if self.is_floating_point else self.dtype.native_type
        ptr: _ffi.CData = unpack_fn(self._ptr)
        if castor is not None:
            ptr = _ffi.cast(f'const {castor}*', ptr)
        unpacked = _ffi.unpack(ptr, self.numel)
        free_fn(ptr)
        return unpacked

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

    def is_shape_eq(self, other: 'Tensor') -> bool:
        return _C.mag_tensor_is_shape_eq(self._ptr, other._ptr)

    def are_strides_eq(self, other: 'Tensor') -> bool:
        return _C.mag_tensor_are_strides_eq(self._ptr, other._ptr)

    def can_broadcast(self, other: 'Tensor') -> bool:
        return _C.mag_tensor_can_broadcast(self._ptr, other._ptr)

    @property
    def is_floating_point(self) -> bool:
        return self.dtype in _FLOATING_POINT_DTYPES

    @property
    def is_integral(self) -> bool:
        return not self.is_floating_point

    @property
    def is_view(self) -> bool:
        return _C.mag_tensor_is_view(self._ptr)

    @property
    def view_base(self) -> Optional['Tensor']:
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
    def is_contiguous(self) -> bool:
        return _C.mag_tensor_is_contiguous(self._ptr)

    @property
    def requires_grad(self) -> bool:
        return _C.mag_tensor_requires_grad(self._ptr)

    @requires_grad.setter
    def requires_grad(self, require: bool) -> None:
        if require and not self.is_floating_point:
            raise RuntimeError(f'Tensors requiring gradients must be of a floating point type, but is: {self.dtype}')
        _C.mag_tensor_set_requires_grad(self._ptr, require)

    @property
    def grad(self) -> Optional['Tensor']:
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

    def clone(self) -> 'Tensor':
        return Tensor(_C.mag_clone(self._ptr))

    def view(self) -> 'Tensor':
        return Tensor(_C.mag_view(self._ptr))

    def transpose(self) -> 'Tensor':
        return Tensor(_C.mag_transpose(self._ptr))

    def detach(self) -> 'Tensor':
        _C.mag_tensor_detach(self._ptr)
        return self

    @property
    def T(self) -> 'Tensor':
        return Tensor(_C.mag_transpose(self._ptr))

    def contiguous(self) -> 'Tensor':
        if self.is_contiguous:
            return self
        return self.clone()

    def permute(self, axes: tuple[int, ...]) -> 'Tensor':
        assert len(axes) == self.rank, f'Invalid number of axes, require {self.rank}, got {len(axes)}'
        if len(axes) != MAX_DIMS:
            axes = axes + tuple(range(self.rank, MAX_DIMS))
        assert len(axes) == MAX_DIMS
        axes = _ffi.new('int64_t[]', axes)
        for i in range(MAX_DIMS):
            assert 0 <= axes[i] < MAX_DIMS
            for j in range(i + 1, MAX_DIMS):
                assert axes[i] != axes[j], f'Duplicate axis: {axes[i]}'
        return Tensor(_C.mag_permute(self._ptr, axes, MAX_DIMS))

    def fill_(self, value: float | int | bool) -> None:
        self._validate_inplace_op()
        if self.is_floating_point:
            _C.mag_tensor_fill_float(self._ptr, float(value))
        else:
            _C.mag_tensor_fill_int(self._ptr, int(value))

    def fill_random_uniform_(self, from_: float | int | None = None, to: float | int | None = None) -> None:
        _validate_dtype_compat(_NUMERIC_DTYPES, self)
        self._validate_inplace_op()
        from_def, to_def = (-0x80000000, 0x7FFFFFFF) if self.is_integral else (0.0, 1.0)
        from_ = from_def if from_ is None else from_
        to = to_def if to is None else to
        assert to > from_, f'Invalid uniform range {to} must be > {from_}'
        if self.is_floating_point:
            _C.mag_tensor_fill_random_uniform_float(self._ptr, float(from_), float(to))
        else:
            _C.mag_tensor_fill_random_uniform_int(self._ptr, int(from_), int(to))

    def fill_random_normal_(self, mean: float, std: float) -> None:
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        self._validate_inplace_op()
        _C.mag_tensor_fill_random_normal(self._ptr, mean, std)

    def fill_random_bernoulli_(self, p: float) -> None:
        _validate_dtype_compat({boolean}, self)
        self._validate_inplace_op()
        _C.mag_tensor_fill_random_bernoulli(self._ptr, p)

    def zeros_(self) -> None:
        self.fill_(0)

    def ones_(self) -> None:
        self.fill_(1)

    def mean(self, dim: int | tuple[int] | None = None, keepdim: bool = False) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        dims, num_dims = _get_reduction_axes(dim)
        return Tensor(_C.mag_mean(self._ptr, dims, num_dims, keepdim))

    def min(self, dim: int | tuple[int] | None = None, keepdim: bool = False) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        dims, num_dims = _get_reduction_axes(dim)
        return Tensor(_C.mag_min(self._ptr, dims, num_dims, keepdim))

    def max(self, dim: int | tuple[int] | None = None, keepdim: bool = False) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        dims, num_dims = _get_reduction_axes(dim)
        return Tensor(_C.mag_max(self._ptr, dims, num_dims, keepdim))

    def sum(self, dim: int | tuple[int] | None = None, keepdim: bool = False) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        dims, num_dims = _get_reduction_axes(dim)
        return Tensor(_C.mag_sum(self._ptr, dims, num_dims, keepdim))

    def argmin(self, dim: int | tuple[int] | None = None, keepdim: bool = False) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        dims, num_dims = _get_reduction_axes(dim)
        return Tensor(_C.mag_argmin(self._ptr, dims, num_dims, keepdim))

    def argmax(self, dim: int | tuple[int] | None = None, keepdim: bool = False) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        dims, num_dims = _get_reduction_axes(dim)
        return Tensor(_C.mag_argmax(self._ptr, dims, num_dims, keepdim))

    def abs(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        return Tensor(_C.mag_abs(self._ptr))

    def abs_(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        self._validate_inplace_op()
        return Tensor(_C.mag_abs_(self._ptr))

    def neg(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        return Tensor(_C.mag_neg(self._ptr))

    def neg_(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        self._validate_inplace_op()
        return Tensor(_C.mag_neg_(self._ptr))

    def __neg__(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        return self.neg()

    def log(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        return Tensor(_C.mag_log(self._ptr))

    def log_(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        self._validate_inplace_op()
        return Tensor(_C.mag_log_(self._ptr))

    def sqr(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        return Tensor(_C.mag_sqr(self._ptr))

    def sqr_(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        self._validate_inplace_op()
        return Tensor(_C.mag_sqr_(self._ptr))

    def sqrt(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        return Tensor(_C.mag_sqrt(self._ptr))

    def sqrt_(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        self._validate_inplace_op()
        return Tensor(_C.mag_sqrt_(self._ptr))

    def sin(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        return Tensor(_C.mag_sin(self._ptr))

    def sin_(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        self._validate_inplace_op()
        return Tensor(_C.mag_sin_(self._ptr))

    def cos(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        return Tensor(_C.mag_cos(self._ptr))

    def cos_(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        self._validate_inplace_op()
        return Tensor(_C.mag_cos_(self._ptr))

    def step(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        return Tensor(_C.mag_step(self._ptr))

    def step_(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        self._validate_inplace_op()
        return Tensor(_C.mag_step_(self._ptr))

    def exp(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        return Tensor(_C.mag_exp(self._ptr))

    def exp_(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        self._validate_inplace_op()
        return Tensor(_C.mag_exp_(self._ptr))

    def floor(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        return Tensor(_C.mag_floor(self._ptr))

    def floor_(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        self._validate_inplace_op()
        return Tensor(_C.mag_floor_(self._ptr))

    def ceil(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        return Tensor(_C.mag_ceil(self._ptr))

    def ceil_(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        self._validate_inplace_op()
        return Tensor(_C.mag_ceil_(self._ptr))

    def round(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        return Tensor(_C.mag_round(self._ptr))

    def round_(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        self._validate_inplace_op()
        return Tensor(_C.mag_round_(self._ptr))

    def softmax(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        return Tensor(_C.mag_softmax(self._ptr))

    def softmax_(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        self._validate_inplace_op()
        return Tensor(_C.mag_softmax_(self._ptr))

    def sigmoid(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        return Tensor(_C.mag_sigmoid(self._ptr))

    def sigmoid_(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        self._validate_inplace_op()
        return Tensor(_C.mag_sigmoid_(self._ptr))

    def hardsigmoid(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        return Tensor(_C.mag_hard_sigmoid(self._ptr))

    def hardsigmoid_(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        self._validate_inplace_op()
        return Tensor(_C.mag_hard_sigmoid_(self._ptr))

    def silu(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        return Tensor(_C.mag_silu(self._ptr))

    def silu_(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        self._validate_inplace_op()
        return Tensor(_C.mag_silu_(self._ptr))

    def tanh(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        return Tensor(_C.mag_tanh(self._ptr))

    def tanh_(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        self._validate_inplace_op()
        return Tensor(_C.mag_tanh_(self._ptr))

    def relu(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        return Tensor(_C.mag_relu(self._ptr))

    def relu_(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        self._validate_inplace_op()
        return Tensor(_C.mag_relu_(self._ptr))

    def gelu(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        return Tensor(_C.mag_gelu(self._ptr))

    def gelu_(self) -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self)
        self._validate_inplace_op()
        return Tensor(_C.mag_gelu_(self._ptr))

    def tril(self, diagonal: int = 0) -> 'Tensor':
        assert self.rank >= 2, f'Tril requires a rank >= 2 but is {self.rank}'
        return Tensor(_C.mag_tril(self._ptr, diagonal))

    def tril_(self, diagonal: int = 0) -> 'Tensor':
        assert self.rank >= 2, f'Tril requires a rank >= 2 but is {self.rank}'
        self._validate_inplace_op()
        return Tensor(_C.mag_tril_(self._ptr, diagonal))

    def triu(self, diagonal: int = 0) -> 'Tensor':
        assert self.rank >= 2, f'Triu requires a rank >= 2 but is {self.rank}'
        return Tensor(_C.mag_triu(self._ptr, diagonal))

    def triu_(self, diagonal: int = 0) -> 'Tensor':
        assert self.rank >= 2, f'Triu requires a rank >= 2 but is {self.rank}'
        self._validate_inplace_op()
        return Tensor(_C.mag_triu_(self._ptr, diagonal))

    def logical_and(self, other: 'Tensor') -> 'Tensor':
        _validate_dtype_compat(_INTEGRAL_DTYPES, self, other)
        return Tensor(_C.mag_and(self._ptr, other._ptr))

    def logical_and_(self, other: 'Tensor') -> 'Tensor':
        _validate_dtype_compat(_INTEGRAL_DTYPES, self, other)
        return Tensor(_C.mag_and_(self._ptr, other._ptr))

    def logical_or(self, other: 'Tensor') -> 'Tensor':
        _validate_dtype_compat(_INTEGRAL_DTYPES, self, other)
        return Tensor(_C.mag_and(self._ptr, other._ptr))

    def logical_or_(self, other: 'Tensor') -> 'Tensor':
        _validate_dtype_compat(_INTEGRAL_DTYPES, self, other)
        return Tensor(_C.mag_or_(self._ptr, other._ptr))

    def logical_xor(self, other: 'Tensor') -> 'Tensor':
        _validate_dtype_compat(_INTEGRAL_DTYPES, self, other)
        return Tensor(_C.mag_and(self._ptr, other._ptr))

    def logical_xor_(self, other: 'Tensor') -> 'Tensor':
        _validate_dtype_compat(_INTEGRAL_DTYPES, self, other)
        return Tensor(_C.mag_xor_(self._ptr, other._ptr))

    def logical_not(self) -> None:
        _validate_dtype_compat(_INTEGRAL_DTYPES, self)
        return Tensor(_C.mag_not(self._ptr))

    def logical_not_(self) -> None:
        _validate_dtype_compat(_INTEGRAL_DTYPES, self)
        self._validate_inplace_op()
        return Tensor(_C.mag_not_(self._ptr))

    def __add__(self, other: object | int | float) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor.full(self.shape, fill_value=float(other))
        _validate_dtype_compat(_NUMERIC_DTYPES, self, other)
        return Tensor(_C.mag_add(self._ptr, other._ptr))

    def __radd__(self, other: int | float) -> 'Tensor':
        other = Tensor.full(self.shape, fill_value=float(other))
        _validate_dtype_compat(_NUMERIC_DTYPES, self, other)
        return other + self

    def __iadd__(self, other: object | int | float) -> 'Tensor':
        self._validate_inplace_op()
        if not isinstance(other, Tensor):
            val: float | int = float(other) if self.is_floating_point else int(other)
            other = Tensor.full(self.shape, dtype=self.dtype, fill_value=val)
        _validate_dtype_compat(_NUMERIC_DTYPES, self, other)
        return Tensor(_C.mag_add_(self._ptr, float(other)))

    def __sub__(self, other: object | int | float) -> 'Tensor':
        if not isinstance(other, Tensor):
            val: float | int = float(other) if self.is_floating_point else int(other)
            other = Tensor.full(self.shape, dtype=self.dtype, fill_value=val)
        _validate_dtype_compat(_NUMERIC_DTYPES, self, other)
        return Tensor(_C.mag_sub(self._ptr, other._ptr))

    def __rsub__(self, other: int | float) -> 'Tensor':
        val: float | int = float(other) if self.is_floating_point else int(other)
        other = Tensor.full(self.shape, dtype=self.dtype, fill_value=val)
        _validate_dtype_compat(_NUMERIC_DTYPES, self, other)
        return other - self

    def __isub__(self, other: object | int | float) -> 'Tensor':
        self._validate_inplace_op()
        if not isinstance(other, Tensor):
            val: float | int = float(other) if self.is_floating_point else int(other)
            other = Tensor.full(self.shape, dtype=self.dtype, fill_value=val)
        _validate_dtype_compat(_NUMERIC_DTYPES, self, other)
        return Tensor(_C.mag_sub_(self._ptr, other._ptr))

    def __mul__(self, other: object | int | float) -> 'Tensor':
        if not isinstance(other, Tensor):
            val: float | int = float(other) if self.is_floating_point else int(other)
            other = Tensor.full(self.shape, dtype=self.dtype, fill_value=val)
        _validate_dtype_compat(_NUMERIC_DTYPES, self, other)
        return Tensor(_C.mag_mul(self._ptr, other._ptr))

    def __rmul__(self, other: int | float) -> 'Tensor':
        val: float | int = float(other) if self.is_floating_point else int(other)
        other = Tensor.full(self.shape, dtype=self.dtype, fill_value=val)
        _validate_dtype_compat(_NUMERIC_DTYPES, self, other)
        return other * self

    def __imul__(self, other: object | int | float) -> 'Tensor':
        self._validate_inplace_op()
        if not isinstance(other, Tensor):
            val: float | int = float(other) if self.is_floating_point else int(other)
            other = Tensor.full(self.shape, dtype=self.dtype, fill_value=val)
        _validate_dtype_compat(_NUMERIC_DTYPES, self, other)
        return Tensor(_C.mag_mul_(self._ptr, other._ptr))

    def __truediv__(self, other: object | int | float) -> 'Tensor':
        if not isinstance(other, Tensor):
            val: float | int = float(other) if self.is_floating_point else int(other)
            other = Tensor.full(self.shape, dtype=self.dtype, fill_value=val)
        _validate_dtype_compat(_NUMERIC_DTYPES, self, other)
        return Tensor(_C.mag_div(self._ptr, other._ptr))

    def __rtruediv__(self, other: int | float) -> 'Tensor':
        val: float | int = float(other) if self.is_floating_point else int(other)
        other = Tensor.full(self.shape, dtype=self.dtype, fill_value=val)
        _validate_dtype_compat(_NUMERIC_DTYPES, self, other)
        return other / self

    def __itruediv__(self, other: object | int | float) -> 'Tensor':
        self._validate_inplace_op()
        if not isinstance(other, Tensor):
            val: float | int = float(other) if self.is_floating_point else int(other)
            other = Tensor.full(self.shape, dtype=self.dtype, fill_value=val)
        _validate_dtype_compat(_NUMERIC_DTYPES, self, other)
        return Tensor(_C.mag_div_(self._ptr, other._ptr))

    def __floordiv__(self, other: object | int | float) -> 'Tensor':
        if not isinstance(other, Tensor):
            val: float | int = float(other) if self.is_floating_point else int(other)
            other = Tensor.full(self.shape, dtype=self.dtype, fill_value=val)
        _validate_dtype_compat(_NUMERIC_DTYPES, self, other)
        return Tensor(_C.mag_div(self._ptr, other._ptr))

    def __rfloordiv__(self, other: int | float) -> 'Tensor':
        val: float | int = float(other) if self.is_floating_point else int(other)
        other = Tensor.full(self.shape, dtype=self.dtype, fill_value=val)
        _validate_dtype_compat(_NUMERIC_DTYPES, self, other)
        return other / self

    def __ifloordiv__(self, other: object | int | float) -> 'Tensor':
        self._validate_inplace_op()
        if not isinstance(other, Tensor):
            val: float | int = float(other) if self.is_floating_point else int(other)
            other = Tensor.full(self.shape, dtype=self.dtype, fill_value=val)
        _validate_dtype_compat(_NUMERIC_DTYPES, self, other)
        return Tensor(_C.mag_div_(self._ptr, other._ptr))

    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self, other)
        return Tensor(_C.mag_matmul(self._ptr, other._ptr))

    def __imatmul__(self, other: 'Tensor') -> 'Tensor':
        _validate_dtype_compat(_FLOATING_POINT_DTYPES, self, other)
        self._validate_inplace_op()
        return Tensor(_C.mag_matmul_(self._ptr, other._ptr))

    def __and__(self, other: object | bool | int) -> 'Tensor':
        if not isinstance(other, Tensor):
            val: bool | int = bool(other) if self.dtype == boolean else int(other)
            other = Tensor.full(self.shape, dtype=self.dtype, fill_value=val)
        _validate_dtype_compat(_INTEGRAL_DTYPES, self, other)
        return Tensor(_C.mag_and(self._ptr, other._ptr))

    def __rand__(self, other: int | bool) -> 'Tensor':
        val: bool | int = bool(other) if self.dtype == boolean else int(other)
        other = Tensor.full(self.shape, dtype=self.dtype, fill_value=val)
        _validate_dtype_compat(_INTEGRAL_DTYPES, self, other)
        return other & self

    def __iand__(self, other: object | int | float) -> 'Tensor':
        self._validate_inplace_op()
        if not isinstance(other, Tensor):
            val: bool | int = bool(other) if self.dtype == boolean else int(other)
            other = Tensor.full(self.shape, dtype=self.dtype, fill_value=val)
        _validate_dtype_compat(_INTEGRAL_DTYPES, self, other)
        return Tensor(_C.mag_and_(self._ptr, other._ptr))

    def __or__(self, other: object | bool | int) -> 'Tensor':
        if not isinstance(other, Tensor):
            val: bool | int = bool(other) if self.dtype == boolean else int(other)
            other = Tensor.full(self.shape, dtype=self.dtype, fill_value=val)
        _validate_dtype_compat(_INTEGRAL_DTYPES, self, other)
        return Tensor(_C.mag_or(self._ptr, other._ptr))

    def __ror__(self, other: int | bool) -> 'Tensor':
        val: bool | int = bool(other) if self.dtype == boolean else int(other)
        other = Tensor.full(self.shape, dtype=self.dtype, fill_value=val)
        _validate_dtype_compat(_INTEGRAL_DTYPES, self, other)
        return other | self

    def __ior__(self, other: object | int | float) -> 'Tensor':
        self._validate_inplace_op()
        if not isinstance(other, Tensor):
            val: bool | int = bool(other) if self.dtype == boolean else int(other)
            other = Tensor.full(self.shape, dtype=self.dtype, fill_value=val)
        _validate_dtype_compat(_INTEGRAL_DTYPES, self, other)
        return Tensor(_C.mag_or_(self._ptr, other._ptr))

    def __xor__(self, other: object | bool | int) -> 'Tensor':
        if not isinstance(other, Tensor):
            val: bool | int = bool(other) if self.dtype == boolean else int(other)
            other = Tensor.full(self.shape, dtype=self.dtype, fill_value=val)
        _validate_dtype_compat(_INTEGRAL_DTYPES, self, other)
        return Tensor(_C.mag_xor(self._ptr, other._ptr))

    def __rxor__(self, other: int | bool) -> 'Tensor':
        val: bool | int = bool(other) if self.dtype == boolean else int(other)
        other = Tensor.full(self.shape, dtype=self.dtype, fill_value=val)
        _validate_dtype_compat(_INTEGRAL_DTYPES, self, other)
        return other ^ self

    def __ixor__(self, other: object | int | float) -> 'Tensor':
        self._validate_inplace_op()
        if not isinstance(other, Tensor):
            val: bool | int = bool(other) if self.dtype == boolean else int(other)
            other = Tensor.full(self.shape, dtype=self.dtype, fill_value=val)
        _validate_dtype_compat(_INTEGRAL_DTYPES, self, other)
        return Tensor(_C.mag_xor_(self._ptr, other._ptr))

    def __invert__(self) -> 'Tensor':
        _validate_dtype_compat(_INTEGRAL_DTYPES, self)
        return Tensor(_C.mag_not(self._ptr))

    def __lshift__(self, other: object | int) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor.full(self.shape, dtype=self.dtype, fill_value=int(other))
        _validate_dtype_compat(_INTEGER_DTYPES, self, other)
        return Tensor(_C.mag_shl(self._ptr, other._ptr))

    def __rlshift__(self, other: int) -> 'Tensor':
        other = Tensor.full(self.shape, dtype=self.dtype, fill_value=int(other))
        _validate_dtype_compat(_INTEGER_DTYPES, self, other)
        return other << self

    def __ilshift__(self, other: object | int) -> 'Tensor':
        self._validate_inplace_op()
        if not isinstance(other, Tensor):
            other = Tensor.full(self.shape, dtype=self.dtype, fill_value=int(other))
        _validate_dtype_compat(_INTEGER_DTYPES, self, other)
        return Tensor(_C.mag_shl_(self._ptr, other._ptr))

    def __rshift__(self, other: object | int) -> 'Tensor':
        if not isinstance(other, Tensor):
            other = Tensor.full(self.shape, dtype=self.dtype, fill_value=int(other))
        _validate_dtype_compat(_INTEGER_DTYPES, self, other)
        return Tensor(_C.mag_shr(self._ptr, other._ptr))

    def __rrshift__(self, other: int) -> 'Tensor':
        other = Tensor.full(self.shape, dtype=self.dtype, fill_value=int(other))
        _validate_dtype_compat(_INTEGER_DTYPES, self, other)
        return other >> self

    def __irshift__(self, other: object | int) -> 'Tensor':
        self._validate_inplace_op()
        if not isinstance(other, Tensor):
            other = Tensor.full(self.shape, dtype=self.dtype, fill_value=int(other))
        _validate_dtype_compat(_INTEGER_DTYPES, self, other)
        return Tensor(_C.mag_shr_(self._ptr, other._ptr))

    def __eq__(self, other: 'Tensor') -> 'Tensor':
        return Tensor(_C.mag_eq(self._ptr, other._ptr))

    def __ne__(self, other: 'Tensor') -> 'Tensor':
        return Tensor(_C.mag_ne(self._ptr, other._ptr))

    def __le__(self, other: 'Tensor') -> 'Tensor':
        _validate_dtype_compat(_NUMERIC_DTYPES, self, other)
        return Tensor(_C.mag_le(self._ptr, other._ptr))

    def __ge__(self, other: 'Tensor') -> 'Tensor':
        _validate_dtype_compat(_NUMERIC_DTYPES, self, other)
        return Tensor(_C.mag_ge(self._ptr, other._ptr))

    def __lt__(self, other: 'Tensor') -> 'Tensor':
        _validate_dtype_compat(_NUMERIC_DTYPES, self, other)
        return Tensor(_C.mag_lt(self._ptr, other._ptr))

    def __gt__(self, other: 'Tensor') -> 'Tensor':
        _validate_dtype_compat(_NUMERIC_DTYPES, self, other)
        return Tensor(_C.mag_gt(self._ptr, other._ptr))

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
        if Context.primary().is_grad_recording and self.requires_grad:
            raise RuntimeError(
                'In-place operations are not allowed when gradient recording is enabled. '
                'Either disable gradient recording or use the `detach()` method to create a new tensor without gradient tracking.'
            )
