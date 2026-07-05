#!/usr/bin/env bash
set -euo pipefail

py-spy record \
  --native \
  --gil \
  --rate 500 \
  -o qwen3.svg \
  -- python examples/qwen3/main.py --prompt "How big is the moon short, short answer?"
