# +---------------------------------------------------------------------+
# | (c) 2025 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations
from functools import lru_cache
from pathlib import Path
from typing import Any
from cffi import FFI
import sys
from importlib.machinery import EXTENSION_SUFFIXES

from magnetron._ffi_cdecl_generated import __MAG_CDECLS


@lru_cache(maxsize=1)
def _load_native_module() -> tuple[FFI, Any]:
    platform = sys.platform
    pkg_dir = Path(__file__).parent

    candidates = []
    if platform.startswith('linux'):
        candidates += ['libmagnetron.so'] + [f'magnetron{ext}' for ext in EXTENSION_SUFFIXES]
    elif platform.startswith('darwin'):
        candidates += ['libmagnetron.dylib'] + [f'magnetron{ext}' for ext in EXTENSION_SUFFIXES]
    elif platform.startswith('win32'):
        candidates += ['magnetron.pyd', 'magnetron.dll']
    else:
        raise RuntimeError(f'Unsupported platform: {platform!r}')

    def find_in_dir(d: Path) -> Path | None:
        for name in candidates:
            p = d / name
            if p.exists():
                return p
        for ext in EXTENSION_SUFFIXES:
            for p in d.glob(f'magnetron*{ext}'):
                return p
        return None

    lib_path = find_in_dir(pkg_dir)
    if lib_path is None:
        for entry in map(Path, sys.path):
            try:
                p = entry / 'magnetron'
                if p.is_dir():
                    hit = find_in_dir(p)
                    if hit:
                        lib_path = hit
                        break
            except Exception:
                pass

    if lib_path is None:
        searched = f'{pkg_dir!r}:{candidates} plus site-packages scan'
        raise FileNotFoundError(f'magnetron shared library not found. Searched: {searched}')

    ffi = FFI()
    ffi.cdef(__MAG_CDECLS)
    lib = ffi.dlopen(str(lib_path))
    return ffi, lib


FFI, C = _load_native_module()
