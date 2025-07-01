# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

import threading
import weakref
from contextlib import ContextDecorator
from functools import lru_cache
from typing import final

from ._bootstrap import load_native_module
from ._core import Config, ComputeDevice, DataType

_ffi, _C = load_native_module()
_MAIN_TID: int = threading.get_native_id()


@final
class Context:
    """Manages the execution context and owns all tensors and active compute devices."""

    @staticmethod
    @lru_cache(maxsize=1)
    def get() -> 'Context':
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


def default_dtype() -> DataType:
    return Context.get().default_dtype


class no_grad(ContextDecorator):
    """Disables gradient recording within a function or block."""

    def __enter__(self) -> None:
        """Disable gradient tracking by stopping the active context's recorder."""
        Context.get().stop_grad_recorder()

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        """Re-enable gradient tracking when exiting the context."""
        Context.get().start_grad_recorder()
