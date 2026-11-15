# +---------------------------------------------------------------------+
# | (c) 2026 Mario Sieg <mario.sieg.64@gmail.com>                       |
# | Licensed under the Apache License, Version 2.0                      |
# |                                                                     |
# | Website : https://mariosieg.com                                     |
# | GitHub  : https://github.com/MarioSieg                              |
# | License : https://www.apache.org/licenses/LICENSE-2.0               |
# +---------------------------------------------------------------------+

from __future__ import annotations

import argparse
import math

from magnetron import Tensor, dtype
from magnetron import distributed as dist
from magnetron.distributed import ReduceOp


def close(a, b, tol=1e-5) -> bool:
    if isinstance(a, list):
        return len(a) == len(b) and all(close(x, y, tol) for x, y in zip(a, b))
    return math.isclose(float(a), float(b), rel_tol=tol, abs_tol=tol)


def check(cond: bool, what: str) -> None:
    if not cond:
        raise AssertionError(what)


def run_collectives(comm: dist.Communicator, r: int, n: int) -> None:
    comm.barrier()

    x = Tensor.full((7,), fill_value=float(r + 1), dtype=dtype.float32)
    comm.broadcast_(x, root=2 % n)
    check(close(x.tolist(), [float(2 % n + 1)] * 7), f'broadcast: {x.tolist()}')

    numel = 3000
    total = n * (n + 1) / 2
    x = Tensor.full((numel,), fill_value=float(r + 1), dtype=dtype.float32)
    comm.all_reduce_(x, ReduceOp.SUM)
    check(close(x.tolist(), [total] * numel), f'all_reduce sum: {x.tolist()[:8]}')

    x = Tensor.full((numel,), fill_value=float(r + 1), dtype=dtype.float32)
    comm.all_reduce_(x, ReduceOp.AVG)
    check(close(x.tolist(), [total / n] * numel), f'all_reduce avg: {x.tolist()[:8]}')

    x = Tensor.full((5,), fill_value=float(r + 1), dtype=dtype.float32)
    comm.all_reduce_(x, ReduceOp.PROD)
    check(close(x.tolist(), [math.factorial(n)] * 5), f'all_reduce prod: {x.tolist()}')

    x = Tensor.full((5,), fill_value=float(r), dtype=dtype.float32)
    comm.all_reduce_(x, ReduceOp.MIN)
    check(close(x.tolist(), [0.0] * 5), f'all_reduce min: {x.tolist()}')
    x = Tensor.full((5,), fill_value=float(r), dtype=dtype.float32)
    comm.all_reduce_(x, ReduceOp.MAX)
    check(close(x.tolist(), [float(n - 1)] * 5), f'all_reduce max: {x.tolist()}')

    x = Tensor.full((33,), fill_value=float(r + 1), dtype=dtype.bfloat16)
    comm.all_reduce_(x, ReduceOp.SUM)
    check(close(x.tolist(), [total] * 33), f'all_reduce bf16: {x.tolist()[:8]}')

    bits = [0b1111, 0b1010, 0b0110, 0b0011]
    want = {ReduceOp.AND: bits[0], ReduceOp.OR: 0, ReduceOp.XOR: 0}
    for i in range(n):
        want[ReduceOp.AND] &= bits[i % 4]
        want[ReduceOp.OR] |= bits[i % 4]
        want[ReduceOp.XOR] ^= bits[i % 4]
    for op, expect in want.items():
        x = Tensor.full((6,), fill_value=bits[r % 4], dtype=dtype.int32)
        comm.all_reduce_(x, op)
        check(x.tolist() == [expect] * 6, f'all_reduce {op}: {x.tolist()} != {expect}')

    x = Tensor.full((9,), fill_value=float(r + 1), dtype=dtype.float32)
    comm.reduce_(x, root=1 % n, op=ReduceOp.SUM)
    if r == 1 % n:
        check(close(x.tolist(), [total] * 9), f'reduce: {x.tolist()}')

    inp = Tensor([float(r * 10 + i) for i in range(5)], dtype=dtype.float32)
    out = Tensor.zeros((n * 5,), dtype=dtype.float32)
    comm.all_gather(out, inp)
    check(close(out.tolist(), [float(j * 10 + i) for j in range(n) for i in range(5)]), f'all_gather: {out.tolist()}')

    inp = Tensor([float(i + r) for i in range(n * 3)], dtype=dtype.float32)
    out = Tensor.zeros((3,), dtype=dtype.float32)
    comm.reduce_scatter(out, inp, ReduceOp.SUM)
    expect = [float(n * (r * 3 + k) + n * (n - 1) / 2) for k in range(3)]
    check(close(out.tolist(), expect), f'reduce_scatter: {out.tolist()} != {expect}')

    inp = Tensor([float(r * 100 + j * 10 + k) for j in range(n) for k in range(2)], dtype=dtype.float32)
    out = Tensor.zeros((n * 2,), dtype=dtype.float32)
    comm.all_to_all(out, inp)
    expect = [float(j * 100 + r * 10 + k) for j in range(n) for k in range(2)]
    check(close(out.tolist(), expect), f'all_to_all: {out.tolist()} != {expect}')

    sc = [(r + j) % 3 + 1 for j in range(n)]
    rc = [(j + r) % 3 + 1 for j in range(n)]
    so = [sum(sc[:j]) for j in range(n)]
    ro = [sum(rc[:j]) for j in range(n)]
    inp = Tensor([float(r * 1000 + j * 100 + k) for j in range(n) for k in range(sc[j])], dtype=dtype.float32)
    out = Tensor.zeros((sum(rc),), dtype=dtype.float32)
    comm.all_to_all_v(out, inp, sc, so, rc, ro)
    expect = [float(j * 1000 + r * 100 + k) for j in range(n) for k in range(rc[j])]
    check(close(out.tolist(), expect), f'all_to_all_v: {out.tolist()} != {expect}')

    if n > 1:
        nxt, prv = (r + 1) % n, (r + n - 1) % n
        sendbuf = Tensor.full((4,), fill_value=float(r), dtype=dtype.float32)
        recvbuf = Tensor.zeros((4,), dtype=dtype.float32)
        if r % 2 == 0:
            comm.send(sendbuf, nxt)
            comm.recv(recvbuf, prv)
        else:
            comm.recv(recvbuf, prv)
            comm.send(sendbuf, nxt)
        check(close(recvbuf.tolist(), [float(prv)] * 4), f'send/recv: {recvbuf.tolist()}')

    x = Tensor.full((4, 6), fill_value=float(r + 1), dtype=dtype.float32)
    v = x[:, 1:4]
    check(not v.is_contiguous, 'view should be non-contiguous')
    comm.all_reduce_(v, ReduceOp.SUM)
    rows = x.tolist()
    for row in rows:
        check(close(row, [r + 1.0, total, total, total, r + 1.0, r + 1.0]), f'view all_reduce: {row}')

    big = 1 << 20
    x = Tensor.full((big,), fill_value=1.0, dtype=dtype.float32)
    comm.all_reduce_(x, ReduceOp.SUM)
    check(close(x[:4].tolist(), [float(n)] * 4) and close(x[big - 4:].tolist(), [float(n)] * 4), 'large all_reduce')

    child = comm.split(r % 2, r)
    check(child is not None and child.size == (n + 1 - r % 2) // 2 and child.rank == r // 2, f'split: {child}')
    x = Tensor.full((3,), fill_value=float(r + 1), dtype=dtype.float32)
    child.all_reduce_(x, ReduceOp.SUM)
    expect = float(sum(i + 1 for i in range(n) if i % 2 == r % 2))
    check(close(x.tolist(), [expect] * 3), f'split all_reduce: {x.tolist()} != {expect}')
    child.destroy()

    rest = comm.split(-1 if r == 0 else 0, r)
    if r == 0:
        check(rest is None, 'rank 0 should not join')
    else:
        check(rest.size == n - 1 and rest.rank == r - 1, f'nocolor split: {rest}')
        rest.barrier()
        rest.destroy()
    comm.barrier()


def run_mesh(ctx: dist.DistributedContext, r: int, n: int) -> None:
    mesh = dist.DeviceMesh(ctx, 'cpu', [[0, 1], [2, 3]], mesh_dim_names=('ep', 'tp'))
    check(mesh.local_rank('ep') == r // 2 and mesh.local_rank('tp') == r % 2, f'mesh coords: {mesh.coords}')
    x = Tensor.full((2,), fill_value=float(r + 1), dtype=dtype.float32)
    mesh.lookup_comm('ep').all_reduce_(x, ReduceOp.SUM)
    expect = float((r % 2 + 1) + (r % 2 + 3))
    check(close(x.tolist(), [expect] * 2), f'mesh ep: {x.tolist()} != {expect}')
    x = Tensor.full((2,), fill_value=float(r + 1), dtype=dtype.float32)
    mesh.lookup_comm('tp').all_reduce_(x, ReduceOp.SUM)
    expect = float((r // 2) * 4 + 3)
    check(close(x.tolist(), [expect] * 2), f'mesh tp: {x.tolist()} != {expect}')
    sub = ctx.create_comm([3, 1])
    if r in (1, 3):
        check(sub is not None and sub.size == 2 and sub.rank == (0 if r == 3 else 1), f'create_comm: {sub}')
    else:
        check(sub is None, 'create_comm on a non-member must return None')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank', type=int, required=True)
    ap.add_argument('--size', type=int, required=True)
    ap.add_argument('--backend', required=True)
    ap.add_argument('--port', type=int, required=True)
    args = ap.parse_args()
    r, n = args.rank, args.size
    comm = dist.Communicator(rank=r, size=n, backend=args.backend, master_addr='127.0.0.1', master_port=args.port, timeout_ms=30000, shm_slot_bytes=4096)
    check(comm.rank == r and comm.size == n and comm.backend == args.backend, f'{comm}')
    run_collectives(comm, r, n)
    comm.destroy()
    if n == 4:
        ctx = dist.DistributedContext(r, n, args.backend, master_addr='127.0.0.1', master_port=args.port, timeout_ms=30000, shm_slot_bytes=4096)
        run_mesh(ctx, r, n)
        ctx.destroy()


if __name__ == '__main__':
    main()
