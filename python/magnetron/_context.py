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
from contextlib import ContextDecorator
from enum import unique, Enum
from functools import lru_cache
from types import TracebackType
from typing import final

from ._bootstrap import FFI, C
from ._core import Config, ComputeDeviceInfo
from ._dtype import DataType

_MAIN_TID: int = threading.get_native_id()


@unique
class PRNGAlgorithm(Enum):
    MERSENNE_TWISTER = C.MAG_PRNG_MERSENNE_TWISTER
    PCG = C.MAG_PRNG_PCG


@final
class Context:
    """Manages the execution context and owns all tensors and active compute devices."""

    def __init__(self, device: ComputeDeviceInfo.CPU | ComputeDeviceInfo.CUDA = Config.compute_device) -> None:
        assert _MAIN_TID == threading.get_native_id(), 'Context must be created in the main thread'
        desc: FFI.CData = FFI.new('mag_device_desc_t*')
        if isinstance(device, ComputeDeviceInfo.CPU):
            desc[0] = C.mag_compute_device_desc_cpu(device.num_threads)
        elif isinstance(device, ComputeDeviceInfo.CUDA):
            desc[0] = C.mag_compute_device_desc_cuda(device.device_id)
        self._ptr = C.mag_ctx_create2(desc)
        self.default_dtype = Config.default_dtype
        self._finalizer = weakref.finalize(self, C.mag_ctx_destroy, self._ptr, True)

    @property
    def native_ptr(self) -> FFI.CData:
        return self._ptr

    @property
    def compute_device_name(self) -> str:
        return FFI.string(C.mag_ctx_get_compute_device_name(self._ptr)).decode('utf-8')

    @property
    def prng_algorithm(self) -> PRNGAlgorithm:
        return PRNGAlgorithm(C.mag_ctx_get_prng_algorithm(self._ptr))

    @prng_algorithm.setter
    def prng_algorithm(self, algorithm: PRNGAlgorithm) -> None:
        C.mag_ctx_set_prng_algorithm(self._ptr, algorithm.value, 0)

    def seed(self, seed: int) -> None:
        C.mag_ctx_set_prng_algorithm(self._ptr, self.prng_algorithm.value, seed)

    @property
    def os_name(self) -> str:
        return FFI.string(C.mag_ctx_get_os_name(self._ptr)).decode('utf-8')

    @property
    def cpu_name(self) -> str:
        return FFI.string(C.mag_ctx_get_cpu_name(self._ptr)).decode('utf-8')

    @property
    def cpu_virtual_cores(self) -> int:
        return C.mag_ctx_get_cpu_virtual_cores(self._ptr)

    @property
    def cpu_physical_cores(self) -> int:
        return C.mag_ctx_get_cpu_physical_cores(self._ptr)

    @property
    def cpu_sockets(self) -> int:
        return C.mag_ctx_get_cpu_sockets(self._ptr)

    @property
    def physical_memory_total(self) -> int:
        return C.mag_ctx_get_physical_memory_total(self._ptr)

    @property
    def physical_memory_free(self) -> int:
        return C.mag_ctx_get_physical_memory_free(self._ptr)

    @property
    def physical_memory_used(self) -> int:
        return abs(self.physical_memory_total - self.physical_memory_free)

    @property
    def is_numa_system(self) -> bool:
        return C.mag_ctx_is_numa_system(self._ptr)

    @property
    def is_profiling(self) -> bool:
        return C.mag_ctx_profiler_is_running(self._ptr)

    def start_grad_recorder(self) -> None:
        C.mag_ctx_grad_recorder_start(self._ptr)

    def stop_grad_recorder(self) -> None:
        C.mag_ctx_grad_recorder_stop(self._ptr)

    @property
    def is_grad_recording(self) -> bool:
        return C.mag_ctx_grad_recorder_is_running(self._ptr)


@lru_cache(maxsize=1)
def active_context() -> 'Context':
    """Get active global context"""
    C.mag_set_log_mode(Config.verbose)
    return Context()


def default_dtype() -> DataType:
    """Get context's default dtype."""
    return active_context().default_dtype


class no_grad(ContextDecorator):
    """Disables gradient recording within a function or block."""

    def __enter__(self) -> None:
        """Disable gradient tracking by stopping the active context's recorder."""
        active_context().stop_grad_recorder()

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        """Re-enable gradient tracking when exiting the context."""
        active_context().start_grad_recorder()
