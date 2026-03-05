# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

__version__ = '0.1.4'
__author__ = 'Mario Sieg'
__email__ = 'mario.sieg.64@gmail.com'
__author_email__ = 'mario.sieg.64@gmail.com'
__license__ = 'Apache 2.0'
__url__ = 'https://github.com/MarioSieg/magnetron'

from contextlib import ContextDecorator
from types import TracebackType

from ._magnetron import *


# __all__ = ['dtype', 'context', 'Tensor', 'Snapshot', 'no_grad']


class no_grad(ContextDecorator):
    """Disables gradient recording within a function or block."""

    def __enter__(self) -> None:
        """Disable gradient tracking by stopping the active context's recorder."""
        context.stop_grad_recorder()

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        """Re-enable gradient tracking when exiting the context."""
        context.start_grad_recorder()
