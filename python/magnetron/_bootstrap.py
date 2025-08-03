# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any
from cffi import FFI
import sys

from magnetron._ffi_cdecl_generated import __MAG_CDECLS


@lru_cache(maxsize=1)
def _load_native_module() -> tuple[FFI, Any]:
    platform = sys.platform
    pkg_dir = Path(__file__).parent  # .../src/magnetron
    root_dir = pkg_dir.parent  # .../src

    # decide which patterns to try
    if platform.startswith('linux'):
        patterns = ['libmagnetron.so', 'magnetron*.so']
    elif platform.startswith('darwin'):
        patterns = ['libmagnetron.dylib', 'magnetron*.so']
    elif platform.startswith('win32'):
        patterns = ['magnetron.dll']
    else:
        raise RuntimeError(f'Unsupported platform: {platform!r}')

    # search in both the package folder and its parent
    lib_path = None
    for search_dir in (pkg_dir, root_dir):
        for pat in patterns:
            hits = list(search_dir.glob(pat))
            if hits:
                lib_path = hits[0]
                break
        if lib_path:
            break

    if not lib_path or not lib_path.exists():
        searched = '; '.join(f'{d!r}:{patterns}' for d in (pkg_dir, root_dir))
        raise FileNotFoundError(f'magnetron shared library not found. Searched: {searched}')

    ffi = FFI()
    ffi.cdef(__MAG_CDECLS)
    lib = ffi.dlopen(str(lib_path))
    return ffi, lib


FFI, C = _load_native_module()
