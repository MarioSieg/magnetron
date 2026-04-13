#!/usr/bin/env bash

rm -f qwen3_profile.nsys-rep
rm -f qwen3_profile.sqlite

nsys profile \
        --trace=cuda,nvtx,osrt \
        --sample=none \
        --cpuctxsw=none \
        --output=qwen3_profile \
        python examples/qwen3/main.py --prompt "Hey" --max_tokens=6

rm -f stats.txt
nsys stats qwen3_profile.nsys-rep >> stats.txt
