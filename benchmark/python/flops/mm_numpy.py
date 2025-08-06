# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import numpy as np
import time

A = np.random.rand(7, 768, 3072).astype(dtype=np.float32)
B = np.random.rand(7, 3072, 768).astype(dtype=np.float32)

batch, M, K = 7, 768, 3072
N = 768
flops = 2 * batch * M * N * K
acc = 0
I = 10
for _ in range(I):
    t0 = time.perf_counter()
    C = A @ B
    t1 = time.perf_counter()
    gflops = flops / (t1 - t0) / 1e9
    print(f"{gflops:.1f}GFLOP/s")
    acc += gflops

print("Average:", acc / 10, "GFLOP/s")
