# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import numpy as np
import time

N = 10000
A = np.random.rand(N//2, N).astype(dtype=np.float32)
B = np.random.rand(N, N//2).astype(dtype=np.float32)

flop = 2 * N**3
avg = 0
I = 10
for _ in range(I):
    st = time.monotonic()
    C = A @ B
    et = time.monotonic()
    s = et - st
    print(f'{flop / s * 1e-12} TFLOP/s')
    avg += flop / s

print(f'Average: {avg / I * 1e-12} TFLOP/s')
