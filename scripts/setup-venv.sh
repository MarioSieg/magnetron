#!/usr/bin/env bash

uv venv
source .venv/bin/activate.fish
uv sync --extra dev
