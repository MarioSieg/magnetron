# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

import argparse
import statistics
import time
from magnetron import Tensor, dtype, distributed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--ip', required=True)
    ap.add_argument('--port', type=int, default=29500)
    ap.add_argument('--rank', type=int, required=True)
    ap.add_argument('--world-size', type=int, default=2)
    ap.add_argument('--numel', type=int, default=16_000_000)
    ap.add_argument('--steps', type=int, default=10_000)
    ap.add_argument('--warmup', type=int, default=5)
    ap.add_argument('--log-every', type=int, default=10)
    args = ap.parse_args()

    pg = distributed.ProcessGroup(args.ip, args.port, args.rank, args.world_size)
    scalar_type = dtype.bfloat16
    x = Tensor.full((args.numel,), fill_value=float(args.rank + 1), dtype=scalar_type)
    expected = args.world_size * (args.world_size + 1) / 2
    tensor_mib = args.numel * scalar_type.size / 1024 / 1024
    wire_mib = tensor_mib * 2
    times = []
    print(
        f'rank {pg.rank}/{pg.world_size}: {args.numel} bf16 elems, tensor={tensor_mib:.2f} MiB, wire≈{wire_mib:.2f} MiB/step',
        flush=True,
    )
    for step in range(1, args.steps + 1):
        x.fill_(float(args.rank + 1))
        t0 = time.perf_counter()
        pg.all_reduce_sum_(x)
        t1 = time.perf_counter()
        dt = t1 - t0
        if step > args.warmup:
            times.append(dt)
        if step % args.log_every == 0:
            first = float(x[0].item())
            ok = abs(first - expected) < 1e-2
            recent = times[-args.log_every :] if len(times) >= args.log_every else times
            avg = statistics.mean(recent) if recent else dt
            bw = wire_mib / avg
            print(
                f'rank {pg.rank} step={step} lat={dt * 1000:.1f} ms avg={avg * 1000:.1f} ms throughput≈{bw:.1f} MiB/s x0={first} ok={ok}',
                flush=True,
            )
    if times:
        avg = statistics.mean(times)
        med = statistics.median(times)
        mn = min(times)
        mx = max(times)
        bw = wire_mib / avg
        print(
            f'rank {pg.rank} summary: '
            f'avg={avg * 1000:.1f} ms '
            f'median={med * 1000:.1f} ms '
            f'min={mn * 1000:.1f} ms '
            f'max={mx * 1000:.1f} ms '
            f'throughput≈{bw:.1f} MiB/s',
            flush=True,
        )


if __name__ == '__main__':
    main()
