import argparse
import os
import time

os.environ.setdefault('OMP_NUM_THREADS', '24')

import numpy as np
import torch
from magnetron import Tensor, dtype

torch.set_num_threads(24)

DT = {
    'float32': (dtype.float32, torch.float32, np.float32),
    'bfloat16': (dtype.bfloat16, torch.bfloat16, None),
    'float16': (dtype.float16, torch.float16, np.float16),
}


def timeit(fn, warmup, iters):
    for _ in range(warmup):
        fn()
    best = float('inf')
    t_all = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        t1 = time.perf_counter()
        t_all.append(t1 - t0)
        best = min(best, t1 - t0)
    t_all.sort()
    return best, t_all[len(t_all) // 2]


def to_np32(t):
    return t.numpy().astype(np.float32)

def run_case(name, shape_a, shape_b, dt_name, warmup, iters, check=True):
    mdtype, tdtype, _ = DT[dt_name]
    A = Tensor.uniform(*shape_a, dtype=mdtype, device='cpu')
    B = Tensor.uniform(*shape_b, dtype=mdtype, device='cpu')

    An, Bn = to_np32(A), to_np32(B)
    ta = torch.from_numpy(An).to(tdtype)
    tb = torch.from_numpy(Bn).to(tdtype)

    mag_best, mag_med = timeit(lambda: A @ B, warmup, iters)
    torch_best, torch_med = timeit(lambda: ta @ tb, warmup, iters)

    # flops
    C = A @ B
    ref = np.matmul(An.astype(np.float32), Bn.astype(np.float32))
    got = to_np32(C)
    flops = 2.0 * np.prod(ref.shape) * shape_a[-1] if ref.ndim else 2.0 * shape_a[-1]
    if ref.ndim == 0:
        flops = 2.0 * shape_a[-1]

    err = ''
    if check:
        denom = np.maximum(np.abs(ref), 1e-3)
        rel = np.abs(got - ref) / denom
        tol = {'float32': 2e-5, 'bfloat16': 6e-2, 'float16': 8e-3}[dt_name]
        p999 = np.quantile(rel, 0.999) if rel.size > 1 else float(rel)
        ok = p999 < tol
        err = 'OK ' if ok else f'FAIL(p999rel={p999:.2e})'

    mag_gf = flops / mag_best / 1e9
    to_gf = flops / torch_best / 1e9
    print(
        f'{name:26s} {dt_name:9s} mag {mag_gf:9.1f} GF/s | torch {to_gf:9.1f} GF/s '
        f'| ratio {mag_gf / to_gf:5.2f}x | {err}'
    )
    return mag_gf, to_gf


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dtypes', default='float32,bfloat16')
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--iters', type=int, default=10)
    p.add_argument('--quick', action='store_true')
    args = p.parse_args()

    cases = [
        ('gemm 256x256x256', (256, 256), (256, 256)),
        ('gemm 512x512x512', (512, 512), (512, 512)),
        ('gemm 1024x1024x1024', (1024, 1024), (1024, 1024)),
        ('gemm 2048x2048x2048', (2048, 2048), (2048, 2048)),
        ('gemm 4096x4096x4096', (4096, 4096), (4096, 4096)),
        ('gemm 1024x4096x1024', (1024, 4096), (4096, 1024)),
        ('gemm skinny 8x4096x4096', (8, 4096), (4096, 4096)),
        ('gemv matvec 4096x4096', (4096, 4096), (4096,)),
        ('gemv vecmat 4096x4096', (4096,), (4096, 4096)),
        ('gemv vecmat 2560x9728', (2560,), (2560, 9728)),
        ('dot 1048576', (1 << 20,), (1 << 20,)),
        ('bmm 32x256x256x256', (32, 256, 256), (32, 256, 256)),
        ('bmm 8x512x512x512', (8, 512, 512), (8, 512, 512)),
    ]
    if args.quick:
        cases = [c for c in cases if '4096x4096x4096' not in c[0]]

    for dt_name in args.dtypes.split(','):
        print(f'--- {dt_name} ---')
        for name, sa, sb in cases:
            try:
                run_case(name, sa, sb, dt_name, args.warmup, args.iters)
            except Exception as e:
                print(f'{name:26s} {dt_name:9s} ERROR {type(e).__name__}: {e}')


if __name__ == '__main__':
    main()
