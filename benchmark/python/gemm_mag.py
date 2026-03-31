# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import time, statistics as stats
import argparse
from magnetron import Tensor, dtype, context

args = argparse.ArgumentParser(description='Benchmark contiguous matmul')
args.add_argument('--B', type=int, default=32)
args.add_argument('--M', type=int, default=512)
args.add_argument('--K', type=int, default=512)
args.add_argument('--N', type=int, default=512)
args.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
args.add_argument('--warmup', type=int, default=20)
args.add_argument('--iters', type=int, default=1000)
args.add_argument('--dtype', type=str, default='bfloat16', choices=['float16', 'bfloat16', 'float32'])
args = args.parse_args()

batch, M, K, N = args.B, args.M, args.K, args.N
A = Tensor.uniform(batch, M, K, dtype=getattr(dtype, args.dtype), device=args.device)
B = Tensor.uniform(batch, N, K, dtype=getattr(dtype, args.dtype), device=args.device)

for _ in range(args.warmup):
    C = A @ B

flops = 2 * batch * M * N * K
times = []
last_result = None
for i in range(args.iters):
    t0 = time.perf_counter()
    C = A @ B
    t1 = time.perf_counter()
    times.append(t1 - t0)
    if i == args.iters-1: # Save last C
        last_result = C
    del C

A, B = A.transfer('cpu'), B.transfer('cpu')
last_result = last_result.transfer('cpu')
errors = ((A @ B) - last_result).abs()
epsilons: dict[str, tuple[float, float]] = {
    'bfloat16': (1.5, 2e-2),
    'float16': (0.5, 1e-2),
    'float32': (1e-4, 1e-4),
}
ref = A @ B
err = (ref - last_result).abs()
atol, rtol = epsilons[args.dtype]
tol = atol + rtol*ref.abs()

if (err > tol).any():
    raise RuntimeError(f"Matmul is wrong (max delta): {err.max().item()}")

gflops = [flops / t / 1e9 for t in times]
print(
    f'Magnetron matmul: median={stats.median(gflops):.1f} GFLOP/s, p10={stats.quantiles(gflops, n=10)[0]:.1f}, p90={stats.quantiles(gflops, n=10)[-1]:.1f}'
)
