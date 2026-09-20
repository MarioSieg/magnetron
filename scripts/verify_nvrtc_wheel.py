"""Compile a tiny kernel with the NVRTC libraries inside each repaired Linux wheel."""

from __future__ import annotations

import ctypes
import sys
import tempfile
import zipfile
from pathlib import Path


def verify(wheel: Path) -> None:
    with tempfile.TemporaryDirectory(prefix='magnetron-nvrtc-') as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(wheel) as archive:
            archive.extractall(root)

        nvrtc = list(root.glob('magnetron.libs/libnvrtc*.so*'))
        nvrtc = [path for path in nvrtc if 'builtins' not in path.name]
        builtins = list(root.glob('magnetron.libs/libnvrtc-builtins.so.*'))
        if len(nvrtc) != 1 or len(builtins) != 1:
            raise RuntimeError(f'{wheel.name}: expected one NVRTC library and its builtins, got {nvrtc}, {builtins}')

        lib = ctypes.CDLL(str(nvrtc[0]))
        lib.nvrtcCreateProgram.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ]
        lib.nvrtcCreateProgram.restype = ctypes.c_int
        lib.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        lib.nvrtcCompileProgram.restype = ctypes.c_int
        lib.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        lib.nvrtcGetProgramLogSize.restype = ctypes.c_int
        lib.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.nvrtcGetProgramLog.restype = ctypes.c_int
        lib.nvrtcGetCUBINSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        lib.nvrtcGetCUBINSize.restype = ctypes.c_int
        lib.nvrtcGetCUBIN.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        lib.nvrtcGetCUBIN.restype = ctypes.c_int
        lib.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        lib.nvrtcDestroyProgram.restype = ctypes.c_int

        program = ctypes.c_void_p()
        source = b'extern "C" __global__ void mag_nvrtc_probe(float *out) { out[0] = 1.0f; }'
        rc = lib.nvrtcCreateProgram(ctypes.byref(program), source, b'probe.cu', 0, None, None)
        if rc != 0:
            raise RuntimeError(f'{wheel.name}: nvrtcCreateProgram failed with code {rc}')
        try:
            options = (ctypes.c_char_p * 1)(b'--gpu-architecture=sm_90')
            rc = lib.nvrtcCompileProgram(program, 1, options)
            if rc != 0:
                size = ctypes.c_size_t()
                lib.nvrtcGetProgramLogSize(program, ctypes.byref(size))
                log = ctypes.create_string_buffer(size.value)
                lib.nvrtcGetProgramLog(program, log)
                raise RuntimeError(f'{wheel.name}: NVRTC compilation failed ({rc}): {log.value.decode()}')
            size = ctypes.c_size_t()
            if lib.nvrtcGetCUBINSize(program, ctypes.byref(size)) != 0 or size.value < 4:
                raise RuntimeError(f'{wheel.name}: NVRTC produced no cubin')
            cubin = ctypes.create_string_buffer(size.value)
            if lib.nvrtcGetCUBIN(program, cubin) != 0 or cubin.raw[:4] != b'\x7fELF':
                raise RuntimeError(f'{wheel.name}: NVRTC produced invalid cubin')
        finally:
            lib.nvrtcDestroyProgram(ctypes.byref(program))
        print(f'{wheel.name}: bundled NVRTC compiled a cubin')


if __name__ == '__main__':
    for name in sys.argv[1:]:
        verify(Path(name))
