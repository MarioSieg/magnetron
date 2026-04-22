# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

**Magnetron** is a compact C99 machine learning runtime with Python bindings (C++17 via nanobind). It is a minimal, hackable alternative to PyTorch with zero runtime dependencies, architecture-aware CPU kernels, and a memory-mapped model format.

## Build & Test Commands

### C++ Library

```sh
# Configure and build
cmake -B build && cmake --build build -j$(nproc)

# Run C++ tests
cd build && ctest --verbose

# Debug build
cmake -B build -DMAGNETRON_DEBUG=ON && cmake --build build -j$(nproc)
```

### Python Bindings & Package

```sh
# Install in editable mode (builds C extension via scikit-build-core)
uv pip install -e ".[test]" -v

# Run Python tests (parallel)
pytest -n 4 -s test/python/

# Run a single test file
pytest -s test/python/test_operators.py

# Run a single test
pytest -s test/python/test_operators.py::test_add
```

### CMake Options

| Flag | Default | Description |
|------|---------|-------------|
| `MAGNETRON_ENABLE_BACKEND_CPU` | ON | CPU backend |
| `MAGNETRON_ENABLE_BACKEND_CUDA` | ON | CUDA backend |
| `MAGNETRON_BUILD_PYTHON_BINDINGS` | ON | nanobind bindings |
| `MAGNETRON_BUILD_TESTS` | ON | C++ unit tests |
| `MAGNETRON_BUILD_BENCHMARKS` | ON | Benchmarks |
| `MAGNETRON_DEBUG` | OFF | Debug mode |

## Architecture

### Layered Design

```
Python API (python/magnetron/)
    └── nanobind Bindings (magnetron/bindings/)
         └── Operator Dispatch (mag_operator.h/c, mag_backend.h/c)
              ├── CPU Backend (magnetron/cpu/) — SIMD-dispatched per microarch
              └── CUDA Backend (magnetron/cuda/) — in progress
                   └── Tensor/Autodiff Core (magnetron/core/)
```

### Core Components (`magnetron/core/`)

- **`mag_tensor.h/c`** — Reference-counted tensors with up to 16 dimensions
- **`mag_dtype.h/c`** — Type system: f32, f16, bf16, bool, int/uint 8/16/32/64
- **`mag_view_solver.h/c`** — View system: slicing, reshaping, broadcasting via shape/stride solving
- **`mag_autodiff.h/c`, `mag_gradients.h/c`** — Dynamic reverse-mode autograd; computation graph built per forward pass
- **`mag_operator.h/c`** — All operator definitions and dispatch table
- **`mag_backend.h/c`** — Backend registration and high-level dispatch
- **`mag_context.h/c`** — Runtime context and thread-local resources
- **`mag_cpuid.c/h`** — CPUID-based microarchitecture detection (selects kernel path at runtime)

### CPU Backend (`magnetron/cpu/`)

Kernels are compiled for 20+ x86-64 microarchitectures (Nehalem through SapphireRapids/Arrowlake) and ARM64 NEON. The correct kernel is selected at runtime via CPUID. SIMD tiers: SSE/SSE2/SSE4, AVX, AVX2+FMA, AVX-512, AVX-512-BF16/FP16.

### Python Layer (`python/magnetron/`)

- `__init__.py` — `Context`, `Tensor`, dtype constants
- `nn/module.py` — `Module`, `Parameter`, `Sequential`
- `nn/layers.py` — `Linear`, `Embedding`, `BatchNorm2D`, `Dropout`, etc.
- `nn/activations.py` — `ReLU`, `GELU`, `LayerNorm`, `Softmax`, etc.
- `nn/loss.py` — `MSELoss`, `CrossEntropyLoss`
- `optim.py` — `SGD`, `Adam` with LR scheduling

### Serialization

Native `.mag` format uses memory-mapped, zero-copy loading (`mag_snapshot_*`). Conversion tools exist for importing external formats.

### Public API

The single public C header is `include/magnetron.h`. Operators are methods on tensors (`x.sin()` style), not free functions.

## Key Conventions

- **C99 core**, **C++17 for bindings and tests only**
- All operators dispatch through the backend table — add new ops in `mag_operator.h/c` and implement in the relevant backend
- CUDA backend is incomplete; CPU backend is the reference implementation
- Python bindings mirror the C API closely; the nanobind glue lives in `magnetron/bindings/`
- Examples in `examples/` demonstrate end-to-end usage: `xor/`, `linear_regression/`, `ae/`, `gpt2/`, `qwen3/`

## Documentation

- `docs/Magnetron-Cheatsheet.md` — Full operator reference with math notation
- `docs/Environment Variables.md` — Runtime configuration via env vars
- `Doxyfile` — Doxygen config for C/C++ API docs
