# (c) 2025 Mario 'Neo' Sieg. <mario.sieg.64@gmail.com>

import time
import statistics as stats
import argparse

import torch


parser = argparse.ArgumentParser(description="Benchmark PyTorch contiguous GEMV-NT path (M=1)")
parser.add_argument("--B", type=int, default=1, help="Batch size")
parser.add_argument("--K", type=int, default=2560, help="Input dimension")
parser.add_argument("--N", type=int, default=9728, help="Output dimension")
parser.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])
parser.add_argument("--warmup", type=int, default=20)
parser.add_argument("--iters", type=int, default=1000)
parser.add_argument("--dtype", type=str, default="bfloat16", choices=["float16", "bfloat16", "float32"])
args = parser.parse_args()

B = args.B
K = args.K
N = args.N

torch_dtype = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}[args.dtype]

device = torch.device(args.device)

# x: [B, 1, K]
# w: [B, K, N]
x = torch.rand((B, 1, K), dtype=torch_dtype, device=device)
w = torch.rand((B, K, N), dtype=torch_dtype, device=device)

if device.type == "cuda":
    torch.cuda.synchronize()

# Warmup
for _ in range(args.warmup):
    y = x @ w

if device.type == "cuda":
    torch.cuda.synchronize()

flops = 2 * B * K * N
times = []
last_result = None

for i in range(args.iters):
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    y = x @ w
    if device.type == "cuda":
        torch.cuda.synchronize()
    t1 = time.perf_counter()

    times.append(t1 - t0)
    if i == args.iters - 1:
        last_result = y
    del y

# Correctness check on CPU
x_cpu = x.float().cpu()
w_cpu = w.float().cpu()
last_result_cpu = last_result.float().cpu()

ref = x_cpu @ w_cpu
err = (ref - last_result_cpu).abs()

epsilons: dict[str, tuple[float, float]] = {
    "bfloat16": (1.5, 2e-2),
    "float16": (0.5, 1e-2),
    "float32": (1e-4, 1e-4),
}

atol, rtol = epsilons[args.dtype]
tol = atol + rtol * ref.abs()

if (err > tol).any():
    raise RuntimeError(f"PyTorch GEMV-NT is wrong (max delta): {err.max().item()}")

gflops = [flops / t / 1e9 for t in times]

print(
    f"PyTorch gemv-nt: "
    f"median={stats.median(gflops):.1f} GFLOP/s, "
    f"p10={stats.quantiles(gflops, n=10)[0]:.1f}, "
    f"p90={stats.quantiles(gflops, n=10)[-1]:.1f}"
)
print(
    f"Shape: B={B}, M=1, K={K}, N={N}, dtype={args.dtype}, device={args.device}"
)
