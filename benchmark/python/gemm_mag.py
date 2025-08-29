# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import time, statistics as stats
import torch
import magnetron as mag

VERIFY = False

batch, M, K, N = 7, 768, 3072, 768
A = mag.Tensor.uniform(batch, M, K)
B = mag.Tensor.uniform(batch, K, N)

for _ in range(10):
    C = A @ B
    _ = C[0, 0, 0].item()

flops = 2 * batch * M * N * K
times = []
results = []
I = 1000
for _ in range(I):
    t0 = time.perf_counter()
    C = A @ B
    s = float(C[0, 0, 0].item())
    t1 = time.perf_counter()
    times.append(t1 - t0)
    results.append(C)

if VERIFY:
    print('Verifying results...')
    correct = torch.tensor(A.tolist(), dtype=torch.float32) @ torch.tensor(B.tolist(), dtype=torch.float32)
    for r in results:
        assert torch.allclose(torch.tensor(r.tolist()), correct, atol=1e-5), 'Results do not match!'

gflops = [flops / t / 1e9 for t in times]
print(
    f'Magnetron matmul: median={stats.median(gflops):.1f} GFLOP/s, p10={stats.quantiles(gflops, n=10)[0]:.1f}, p90={stats.quantiles(gflops, n=10)[-1]:.1f}'
)
