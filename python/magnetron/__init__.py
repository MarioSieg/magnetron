# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from contextlib import ContextDecorator
from types import TracebackType

from . import _magnetron_bindings
from ._magnetron_bindings import *

__version__ = _magnetron_bindings.__version__
__snapshot_version__ = _magnetron_bindings.__snapshot_version__
__author__ = _magnetron_bindings.__author__
__email__ = _magnetron_bindings.__email__
__author_email__ = _magnetron_bindings.__author_email__
__license__ = _magnetron_bindings.__license__
__url__ = _magnetron_bindings.__url__

from contextlib import ContextDecorator
from types import TracebackType


class device(ContextDecorator):
    """Sets the default device within a function or block."""

    def __init__(self, device_name: str) -> None:
        self.device_name = device_name
        self.prev_dev: str | None = None

    def __enter__(self) -> None:
        self.prev_dev = context.get_default_device()
        if not context.is_device_available(self.device_name):
            raise RuntimeError(f'Requested device {self.device_name} not available')
        context.set_default_device(self.device_name)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        assert self.prev_dev is not None
        context.set_default_device(self.prev_dev)


class no_grad(ContextDecorator):
    """Disables gradient recording within a function or block."""

    def __init__(self) -> None:
        # A ContextDecorator instance is shared by every call of the function it decorates,
        # so the saved state must be a stack to survive nesting and recursion.
        self.prev_recording: list[bool] = []

    def __enter__(self) -> None:
        """Disable gradient tracking by stopping the active context's recorder."""
        self.prev_recording.append(context.is_grad_recording())
        context.stop_grad_recorder()

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        """Restore whatever gradient tracking state was active on entry."""
        if self.prev_recording.pop():
            context.start_grad_recorder()


class fuse(ContextDecorator):
    """Fuse the pointwise operations in this block into generated kernels.

    Operations that can join a chain are recorded instead of executed, and the chain is compiled
    into a single pass when the block ends or when something reads a value it has not produced yet.
    Anything that cannot join - a matmul, a reduction, a strided operand - flushes the chain and
    then runs normally, so a block may contain anything. Results are unchanged either way; with the
    JIT off or no host compiler present the recorded ops just run in order.

        with no_grad(), mag.fuse():
            y = x * w + b
            z = y * y - x

    Use it under no_grad, where an intermediate nothing else refers to never has to reach memory:
    on an 8-operation float32 chain, 1.8x at 4K elements rising to 6x at 4M on an Apple M3. With
    gradient recording on it is roughly neutral, because the autograd graph holds every intermediate
    for backward and only the loads are saved.

    Transcendentals end a chain instead of joining it. Their CPU kernels are approximations, and
    generated libm calls would not reproduce them bit for bit.
    """

    def __enter__(self) -> None:
        context.begin_fusion()

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        context.end_fusion()
