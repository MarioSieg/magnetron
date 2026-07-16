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
