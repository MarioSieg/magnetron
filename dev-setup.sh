#!/usr/bin/env bash
set -euo pipefail

uv venv
source .venv/bin/activate.fish
uv sync --extra dev
