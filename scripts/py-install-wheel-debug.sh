#!/usr/bin/env bash
set -euo pipefail

SCRIPT="$1"
shift
uv pip install -e . -C cmake.define.CMAKE_BUILD_TYPE=Debug
valgrind \
  --leak-check=full \
  --show-leak-kinds=all \
  --track-origins=yes \
  --error-exitcode=1 \
  uv run python "$SCRIPT" "$@"
