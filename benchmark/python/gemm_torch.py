# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import os, time, statistics as stats, multiprocessing as mp

nthreads = mp.cpu_count()
os.environ['OMP_NUM_THREADS'] = str(nthreads)
os.environ['MKL_NUM_THREADS'] = str(nthreads)

import torch

torch.set_num_threads(nthreads)
torch.set_grad_enabled(False)

batch, M, K, N = 7, 768, 3072, 768
A = torch.rand(batch, M, K, dtype=torch.float32)
B = torch.rand(batch, K, N, dtype=torch.float32)

for _ in range(10):
    C = A @ B
    _ = float(C[0, 0, 0].item())

flops = 2 * batch * M * N * K
times = []
I = 1000
for _ in range(I):
    t0 = time.perf_counter()
    C = A @ B
    _ = float(C[0, 0, 0].item())
    t1 = time.perf_counter()
    times.append(t1 - t0)

gflops = [flops / t / 1e9 for t in times]
print(
    f'Torch matmul: median={stats.median(gflops):.1f} GFLOP/s, p10={stats.quantiles(gflops, n=10)[0]:.1f}, p90={stats.quantiles(gflops, n=10)[-1]:.1f}'
)
