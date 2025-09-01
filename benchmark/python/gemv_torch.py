# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

import multiprocessing
import torch
import time

torch.set_num_threads(multiprocessing.cpu_count())

A = torch.rand(1, 1, 768, dtype=torch.float32)
B = torch.rand(768, 50257, dtype=torch.float32)

batch, M, K = 1, 1, 768
N = 768
flops = 2 * batch * M * N * K
acc = 0
I = 1000
for _ in range(I):
    t0 = time.perf_counter()
    C = A @ B
    t1 = time.perf_counter()
    gflops = flops / (t1 - t0) / 1e9
    acc += gflops

print('Average:', acc / I, 'GFLOP/s')
