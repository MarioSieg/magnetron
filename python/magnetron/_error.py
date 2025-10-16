# +---------------------------------------------------------------------+
# | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

from ._context import active_context, NativeErrorInfo
from ._bootstrap import C, FFI


class MagnetronError(RuntimeError):
    def __init__(self, message: str, native_info: NativeErrorInfo | None) -> None:
        super().__init__(message)
        self.native_info = native_info


def _handle_errc(status: int) -> None:
    if status == C.MAG_STATUS_OK:
        return
    ctx = active_context()
    error_info: NativeErrorInfo | None = ctx.take_last_error()
    ercc_name: str = FFI.string(C.mag_status_get_name(status)).decode('utf-8')
    msg = f'Magnetron C runtime error: #0x{status:08X} ({ercc_name})\n'
    if error_info is not None:
        msg += f'{error_info.message} (triggered at {error_info.file}:{error_info.line})'
    raise MagnetronError(msg, error_info)
