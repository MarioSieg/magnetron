#!/usr/bin/env bash
set -euo pipefail

uv pip install -e . -C cmake.define.CMAKE_BUILD_TYPE=Debug
