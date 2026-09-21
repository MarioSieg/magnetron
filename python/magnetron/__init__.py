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

import threading
from contextlib import ContextDecorator
from types import TracebackType


def _saved_state_stack(tls: threading.local) -> list:
    stack = getattr(tls, 'stack', None)
    if stack is None:
        stack = []
        tls.stack = stack
    return stack


class device(ContextDecorator):
    """Sets the default device within a function or block. Applies to the calling thread only."""

    _tls = threading.local()

    def __init__(self, device_name: str) -> None:
        self.device_name = device_name

    def __enter__(self) -> None:
        if not context.is_device_available(self.device_name):
            raise RuntimeError(f'Requested device {self.device_name} not available')
        _saved_state_stack(self._tls).append(context.get_default_device())
        context.set_default_device(self.device_name)

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        context.set_default_device(_saved_state_stack(self._tls).pop())


class no_grad(ContextDecorator):
    """Disables gradient recording within a function or block. Applies to the calling thread only."""

    _tls = threading.local()

    def __enter__(self) -> None:
        """Disable gradient tracking by stopping the active context's recorder."""
        _saved_state_stack(self._tls).append(context.is_grad_recording())
        context.stop_grad_recorder()

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        """Restore whatever gradient tracking state was active on entry."""
        if _saved_state_stack(self._tls).pop():
            context.start_grad_recorder()

class fuse(ContextDecorator):
    """Record fusible pointwise operators and run them as one chain.

    Inside the region an operator that can join the chain is recorded rather than executed, and its
    result is not computed until something needs it: leaving the region, reading the values, or an
    operator arriving that cannot join. The answers are identical to running eagerly, bit for bit.

    Regions nest, and only leaving the outermost one runs anything, so a helper that opens a region
    can be called from code that already did without cutting the chain in half.
    A context accepts one active region owner at a time; another thread gets an error rather than
    mixing its operations into that chain.

        with no_grad(), fuse():
            y = x * w + b
            z = y * y - x
    """

    def __enter__(self) -> None:
        context.begin_fusion()

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        context.end_fusion()


def fusion_stats() -> dict[str, int]:
    """How many chains have run and how many operators went into them."""
    return context.fusion_stats()
