# (c) 2026 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import time, statistics as stats
import argparse
import math
from magnetron import Tensor, dtype

args = argparse.ArgumentParser(description='Benchmark unary op')
args.add_argument('op', type=str, help='Unary op name, e.g. exp, tanh, sigmoid, relu')
args.add_argument('--N', type=int, default=1 << 24)
args.add_argument('--device', type=str, default='cpu', choices=['cpu', 'cuda'])
args.add_argument('--warmup', type=int, default=20)
args.add_argument('--iters', type=int, default=500)
args.add_argument('--dtype', type=str, default='bfloat16', choices=['float8_e4m3fn', 'float16', 'bfloat16', 'float32'])
args.add_argument('--strided', action='store_true', help='Use non-contiguous tensors')
args = args.parse_args()

_dtype = getattr(dtype, args.dtype)

cbrt = int(math.ceil(math.cbrt(args.N)))
x = Tensor.uniform(cbrt, cbrt, cbrt, dtype=_dtype, device=args.device)
if args.strided:
    x = x.permute(*reversed([i for i in range(x.rank)]))
    assert not x.is_contiguous
print(x.shape)
op = getattr(x, args.op)

for _ in range(args.warmup):
    y = op()

times = []
for _ in range(args.iters):
    t0 = time.perf_counter()
    y = op()
    t1 = time.perf_counter()
    times.append(t1 - t0)
    del y

gb = (args.N * _dtype.size * 2) / 1e9
throughput = [gb / t for t in times]

print(
    f'Magnetron {args.op}: median={stats.median(throughput):.2f} GB/s, '
    f'p10={stats.quantiles(throughput, n=10)[0]:.2f}, '
    f'p90={stats.quantiles(throughput, n=10)[-1]:.2f}'
)
