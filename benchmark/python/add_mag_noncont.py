# (c) 2025 Mario Sieg. <mario.sieg.64@gmail.com>

import magnetron as mag
import time

A = mag.Tensor.uniform(1024, 1024)
B = mag.Tensor.uniform(1024, 1024).permute(1, 0)

batch, M, K = 256, 256, 256
N = 256
flops = 2 * batch * M * N * K
acc = 0
I = 100
for _ in range(I):
    t0 = time.perf_counter()
    C = A + B
    t1 = time.perf_counter()
    gflops = flops / (t1 - t0) / 1e9
    print(f'{gflops:.1f}GFLOP/s')
    acc += gflops

print('Average:', acc / I, 'GFLOP/s')
