# +---------------------------------------------------------------------+
# | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from . import context
from ._bootstrap import _C, _FFI


class MagnetronError(RuntimeError):
    def __init__(self, message: str, native_info: context.NativeErrorInfo | None) -> None:
        super().__init__(message)
        self.native_info = native_info


def _handle_errc(status: int) -> None:
    if status == _C.MAG_STATUS_OK:
        return
    info: context.NativeErrorInfo | None = context.take_last_error()
    ercc_name: str = _FFI.string(_C.mag_status_get_name(status)).decode('utf-8')
    msg = f'Magnetron C runtime error: 0x{status:X} ({ercc_name})\n'
    if info is not None:
        msg += f'Triggered at {info.file}:{info.line}\n\n'
        msg += f'{info.message}\n'
    raise MagnetronError(msg, info)
