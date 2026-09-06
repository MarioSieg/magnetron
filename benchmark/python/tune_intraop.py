"""Find the element count at which intra-op multithreading starts paying off.

The per-op thresholds in magnetron/cpu/mag_cpu_autotune.c decide when an op is spread across
the threadpool. Below the crossover, fan-out and barrier cost more than the work they save.

This runs each op single-threaded and threaded at a range of sizes and reports the smallest size
where threading actually wins, so the table can be re-tuned on a new machine instead of guessed.
Each configuration runs in its own process because MAG_CPU_INTRAOP_MIN_ELEMS is read once.

What it reports is a ratio between two runs of the same op at the same size, so the per-call cost of
getting from Python into the kernel appears in both and cancels out of the direction of the answer.
It does pull the ratio toward 1.0 at small sizes, which errs toward leaving an op single-threaded.
Treat the crossover as a place to start, not a precise figure.

Run:
    python benchmark/python/tune_intraop.py
    python benchmark/python/tune_intraop.py --ops ADD MUL EXP --repeats 300
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Callable

from magnetron import Tensor, dtype

FORCE_SINGLE = 10**9  # Any size below this stays on the calling thread.
FORCE_THREADED = 0  # Every size goes to the threadpool.

SIZES = [4096, 16384, 65536, 262144, 1048576, 4194304]

# One representative per cost class: cheap bandwidth-bound binaries, costlier elementwise,
# a pure write, a copy, a reduction, and an expensive compound activation.
OPS = {
    'ADD': ('binary', lambda a, b: a + b),
    'MUL': ('binary', lambda a, b: a * b),
    'DIV': ('binary', lambda a, b: a / b),
    'SUB': ('binary', lambda a, b: a - b),
    'SQRT': ('unary', lambda a: a.sqrt()),
    'EXP': ('unary', lambda a: a.exp()),
    'TANH': ('unary', lambda a: a.tanh()),
    'GELU': ('unary', lambda a: a.gelu()),
    'SIGMOID': ('unary', lambda a: a.sigmoid()),
    'SQR': ('unary', lambda a: a.sqr()),
    'RELU': ('unary', lambda a: a.relu()),
    'SILU': ('unary', lambda a: a.silu()),
    'SOFTMAX': ('unary', lambda a: a.softmax()),
    'GELU_APPROX': ('unary', lambda a: a.gelu_approx()),
    'ABS': ('unary', lambda a: a.abs()),
    'LOG': ('unary', lambda a: a.log()),
    'NEG': ('unary', lambda a: a.neg()),
    'FILL': ('unary', lambda a: a.fill_(1.0)),
    'CAST': ('unary', lambda a: a.cast(dtype.float16)),
    'CLONE': ('unary', lambda a: a.clone()),
    'SUM': ('unary', lambda a: a.sum()),
    'MEAN': ('unary', lambda a: a.mean()),
}


def _time_calls(fn: Callable[[], object], repeats: int) -> int:
    """Nanoseconds per call, wall clock, including the Python and binding cost of making the call."""
    for _ in range(max(8, repeats // 10)):  # warm the kernel dispatch tables and the allocator
        fn()
    start = time.perf_counter_ns()
    for _ in range(repeats):
        fn()
    return (time.perf_counter_ns() - start) // repeats


def _measure(op: str, size: int, repeats: int) -> int:
    """Nanoseconds per call, wall clock, including the cost of getting from Python into the kernel.

    That overhead is not subtracted, deliberately. It is the same in both runs being compared, so
    it cancels out of the direction of the answer and only pulls the ratio toward 1.0 - which at
    the small end, where the overhead is the same order as the kernel, means the tool understates
    the benefit of threading, never invents one. Subtracting an estimate of it instead
    leaves a difference of two nearly equal numbers, and the ratio turns to noise.
    """
    kind, fn = OPS[op]
    a = Tensor.uniform((size,), low=1.0, high=2.0)
    b = Tensor.uniform((size,), low=1.0, high=2.0)
    return _time_calls((lambda: fn(a, b)) if kind == 'binary' else (lambda: fn(a)), repeats)


def _worker() -> None:
    args = json.loads(sys.argv[2])
    out = {str(size): _measure(args['op'], size, args['repeats']) for size in args['sizes']}
    print('RESULT ' + json.dumps(out))


def _run_child(op: str, sizes: list[int], repeats: int, threshold: int) -> dict[int, int]:
    env = dict(os.environ)
    env['MAG_CPU_INTRAOP_MIN_ELEMS'] = str(threshold)
    env['MAG_LOG_LEVEL'] = 'off'
    payload = json.dumps({'op': op, 'sizes': sizes, 'repeats': repeats})
    proc = subprocess.run([sys.executable, __file__, '--worker', payload], env=env, capture_output=True, text=True)
    for line in proc.stdout.splitlines():
        if line.startswith('RESULT '):
            return {int(k): v for k, v in json.loads(line[7:]).items()}
    raise RuntimeError(f'worker for {op} produced no result:\n{proc.stdout}\n{proc.stderr}')


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--ops', nargs='*', default=list(OPS), choices=list(OPS))
    ap.add_argument('--sizes', nargs='*', type=int, default=SIZES)
    ap.add_argument('--repeats', type=int, default=200)
    args = ap.parse_args()

    print()
    print(f'{"op":<10}' + ''.join(f'{s:>13}' for s in args.sizes) + f'{"crossover":>14}')
    print(f'{"":<10}' + ''.join(f'{"(1t/Nt)":>13}' for _ in args.sizes))
    print('-' * (10 + 13 * len(args.sizes) + 14))

    recommendations: dict[str, int | None] = {}
    for op in args.ops:
        reps = args.repeats if max(args.sizes) <= 1048576 else max(20, args.repeats // 8)
        single = _run_child(op, args.sizes, reps, FORCE_SINGLE)
        threaded = _run_child(op, args.sizes, reps, FORCE_THREADED)
        cells = []
        crossover = None
        for size in args.sizes:
            s, t = single[size], threaded[size]
            if s <= 0 or t <= 0:
                cells.append(f'{"-":>13}')
                continue
            ratio = s / t  # >1 means threading wins
            cells.append(f'{ratio:>12.2f}x')
            if crossover is None and ratio > 1.15:  # needs a real margin, not noise
                crossover = size
        recommendations[op] = crossover
        label = f'{crossover:,}' if crossover else 'never wins'
        print(f'{op:<10}' + ''.join(cells) + f'{label:>14}')

    print()
    print('Ratio is single-threaded time divided by threaded time. Above 1.00 means threading helps.')
    print('Crossover is the smallest measured size where threading wins by more than 15%.')
    print()
    print('Suggested thread_treshold values for mag_cpu_autotune.c:')
    for op, crossover in recommendations.items():
        if crossover is None:
            print(f'  MAG_OP_{op:<10} threading never wins at these sizes; keep it single-threaded')
        else:
            print(f'  MAG_OP_{op:<10} {crossover}')
    print()


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--worker':
        _worker()
    else:
        main()
