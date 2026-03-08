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


class no_grad(ContextDecorator):
    """Disables gradient recording within a function or block."""

    def __enter__(self) -> None:
        """Disable gradient tracking by stopping the active context's recorder."""
        context.stop_grad_recorder()

    def __exit__(self, exc_type: type[BaseException] | None, exc_value: BaseException | None, traceback: TracebackType | None) -> None:
        """Re-enable gradient tracking when exiting the context."""
        context.start_grad_recorder()
