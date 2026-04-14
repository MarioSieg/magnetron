sudo "$(which ncu)" "$(which python)" benchmark/python/gemm_mag.py --device=cuda --warmup=3 --iters=5
