# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

import multiprocessing
import torch
import time

torch.set_num_threads(multiprocessing.cpu_count())

A = torch.rand(7, 3072, 768)
B = torch.rand(7, 3072, 768)

batch, M, K = 7, 3072, 768
N = 768
flops = 2 * batch * M * N * K
acc = 0
I = 1000
for _ in range(I):
    t0 = time.perf_counter()
    C = A + B
    t1 = time.perf_counter()
    gflops = flops / (t1 - t0) / 1e9
    print(f'{gflops:.1f}GFLOP/s')
    acc += gflops

print('Average:', acc / I, 'GFLOP/s')
