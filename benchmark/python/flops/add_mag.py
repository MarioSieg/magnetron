# (c) 2025 Mario "Neo" Sieg. <mario.sieg.64@gmail.com>

import magnetron as mag
import time


N = 2048
A = mag.Tensor.uniform((N, N))
B = mag.Tensor.uniform((N, N))
print(A.shape)

flop = 2 * N**3
avg = 0
I = 1000
for _ in range(I):
    st = time.monotonic()
    C = A + B
    et = time.monotonic()
    s = et - st
    print(f'{flop / s * 1e-12} TFLOP/s')
    avg += flop / s

print(f'Average: {avg / I * 1e-12} TFLOP/s')
