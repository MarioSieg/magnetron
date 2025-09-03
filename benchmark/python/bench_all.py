# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

from bench import *

DIM_LIM: int = 4096
STEP: int = 64

bench_square_bin_ops(dim_lim=DIM_LIM, step=STEP)
bench_square_matmul(dim_lim=DIM_LIM, step=STEP)
bench_permuted_bin_ops(dim_lim=DIM_LIM, step=STEP)
bench_permuted_matmul(dim_lim=DIM_LIM, step=STEP)
