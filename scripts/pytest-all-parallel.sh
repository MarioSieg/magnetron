#!/usr/bin/env bash
set -e
cores=$( (command -v nproc >/dev/null && nproc) || sysctl -n hw.ncpu )
div=$([ "$(uname)" = Darwin ] && echo 2 || echo 4)
workers=$(( cores / div )); [ $workers -lt 1 ] && workers=1
(command -v uv >/dev/null && uv pip install . --force-reinstall) || pip install . --force-reinstall
pytest -n "$workers" -s test/python/
