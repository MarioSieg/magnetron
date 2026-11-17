# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from . import _magnetron_bindings
from ._magnetron_bindings import (
    SnapshotStreamReader as SnapshotStreamReader,
    SnapshotStreamWriter as SnapshotStreamWriter,
    Tensor as Tensor,
    context as context,
    dtype as dtype,
    sync as sync,
)

__all__ = [
    'SnapshotStreamReader',
    'SnapshotStreamWriter',
    'Tensor',
    'context',
    'device',
    'distributed',
    'dtype',
    'nn',
    'no_grad',
    'optim',
    'snapshot',
    'sync',
    'utils',
]

__version__ = _magnetron_bindings.__version__
__snapshot_version__ = _magnetron_bindings.__snapshot_version__
__author__ = _magnetron_bindings.__author__
__email__ = _magnetron_bindings.__email__
__author_email__ = _magnetron_bindings.__author_email__
__license__ = _magnetron_bindings.__license__
__url__ = _magnetron_bindings.__url__

import functools
import inspect
import threading
from collections.abc import Callable, Generator
from contextlib import ContextDecorator
from types import TracebackType
from typing import Any


def _saved_state_stack(tls: threading.local) -> list:
    stack = getattr(tls, 'stack', None)
    if stack is None:
        stack = []
        tls.stack = stack
    return stack


class _StateDecorator(ContextDecorator):
    def __call__(self, func: Callable[..., Any]) -> Callable[..., Any]:
        if not inspect.isgeneratorfunction(func):
            return super().__call__(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Generator[Any, Any, Any]:
            gen = func(*args, **kwargs)
            try:
                with self._recreate_cm():
                    item = gen.send(None)
                while True:
                    try:
                        sent = yield item
                    except GeneratorExit:
                        with self._recreate_cm():
                            gen.close()
                        raise
                    except BaseException as e:
                        with self._recreate_cm():
                            item = gen.throw(e)
                    else:
                        with self._recreate_cm():
                            item = gen.send(sent)
            except StopIteration as stop:
                return stop.value

        return wrapper


class device(_StateDecorator):
    _tls = threading.local()

    def __init__(self, device_name: str) -> None:
        self.device_name = device_name

    def __enter__(self) -> None:
        if not context.is_device_available(self.device_name):
            raise RuntimeError(f'Requested device {self.device_name} not available')
        _saved_state_stack(self._tls).append(context.get_default_device())
        context.set_default_device(self.device_name)

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        context.set_default_device(_saved_state_stack(self._tls).pop())


class no_grad(_StateDecorator):
    _tls = threading.local()

    def __enter__(self) -> None:
        _saved_state_stack(self._tls).append(context.is_grad_recording())
        context.stop_grad_recorder()

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        if _saved_state_stack(self._tls).pop():
            context.start_grad_recorder()


from . import distributed as distributed, nn as nn, optim as optim, snapshot as snapshot  # noqa: E402
