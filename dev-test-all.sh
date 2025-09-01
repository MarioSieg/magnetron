#!/usr/bin/env bash

# Must be in venv already with .[dev] installed
uv pip install . --force-reinstall
# Use half of CPUs as the kernels themselves need cores too and if we dispatch too many the CPU gets overloaded alot with dispatching work and working
num_cores=$(($(nproc) / 4))
pytest -n "$num_cores" -s test/python/
