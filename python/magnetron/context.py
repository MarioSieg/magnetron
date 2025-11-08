# +---------------------------------------------------------------------+
# | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import weakref
from contextlib import ContextDecorator
from dataclasses import dataclass
from types import TracebackType

from ._bootstrap import _FFI, _C


@dataclass
class NativeErrorInfo:
    status_code: int
    message: str
    file: str
    line: int
    col: int
    func: str


_ctx = _C.mag_ctx_create(bytes('cpu', 'utf-8'))
_ctx_finalizer = weakref.finalize(_ctx, _C.mag_ctx_destroy, _ctx, True)


def native_ptr() -> _FFI.CData:
    return _ctx


def last_error_code() -> int:
    return _C.mag_ctx_get_last_error_code(_ctx)


def has_last_error() -> bool:
    return _C.mag_ctx_has_error(_ctx)


def clear_last_error() -> None:
    _C.mag_ctx_clear_last_error(_ctx)


def last_error_str() -> str:
    return _C.mag_status_get_name(_C.mag_ctx_get_last_error_code(_ctx)).decode('utf-8')


def take_last_error() -> NativeErrorInfo | None:
    if not has_last_error():
        return None
    native_err: _FFI.CData = _FFI.new('mag_error_t*')
    _C.mag_ctx_take_last_error(_ctx, native_err)
    status_code: int = native_err.code
    message: str = _FFI.string(native_err.message).decode('utf-8')
    file: str = _FFI.string(native_err.file).decode('utf-8')
    line: int = native_err.line
    col: int = native_err.col
    func: str = _FFI.string(native_err.func).decode('utf-8')
    return NativeErrorInfo(status_code, message, file, line, col, func)


def start_grad_recorder() -> None:
    _C.mag_ctx_grad_recorder_start(_ctx)


def stop_grad_recorder() -> None:
    _C.mag_ctx_grad_recorder_stop(_ctx)


def is_grad_recording() -> bool:
    return _C.mag_ctx_grad_recorder_is_running(_ctx)


def manual_seed(seed: int) -> None:
    if seed < 0:
        seed += 0xFFFFFFFFFFFFFFFF
    seed &= 0xFFFFFFFFFFFFFFFF
    _C.mag_ctx_manual_seed(_ctx, seed)


def compute_device_name() -> str:
    return _FFI.string(_C.mag_ctx_get_compute_device_name(_ctx)).decode('utf-8')


def os_name() -> str:
    return _FFI.string(_C.mag_ctx_get_os_name(_ctx)).decode('utf-8')


def cpu_name() -> str:
    return _FFI.string(_C.mag_ctx_get_cpu_name(_ctx)).decode('utf-8')


def cpu_virtual_cores() -> int:
    return _C.mag_ctx_get_cpu_virtual_cores(_ctx)


def cpu_physical_cores() -> int:
    return _C.mag_ctx_get_cpu_physical_cores(_ctx)


def cpu_sockets() -> int:
    return _C.mag_ctx_get_cpu_sockets(_ctx)


def physical_memory_total() -> int:
    return _C.mag_ctx_get_physical_memory_total(_ctx)


def physical_memory_free() -> int:
    return _C.mag_ctx_get_physical_memory_free(_ctx)


def physical_memory_used() -> int:
    return abs(physical_memory_total() - physical_memory_free())


def is_numa_system() -> bool:
    return _C.mag_ctx_is_numa_system(_ctx)


class no_grad(ContextDecorator):
    """Disables gradient recording within a function or block."""

    def __enter__(self) -> None:
        """Disable gradient tracking by stopping the active context's recorder."""
        stop_grad_recorder()

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        """Re-enable gradient tracking when exiting the context."""
        start_grad_recorder()
